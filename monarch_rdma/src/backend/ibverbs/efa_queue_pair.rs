/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! EFA SRD queue pair built on the efadv and extended-verbs work-request
//! builder.

use std::io::Error;
use std::result::Result;
use std::sync::Arc;

use super::domain::IbvDomain;
use super::domain::IbvDomainImpl;
use super::memory_region::IbvMemoryRegionView;
use super::memory_region::IbvRemoteMemoryRegionView;
use super::primitives::Gid;
use super::primitives::IbvAh;
use super::primitives::IbvConfig;
use super::primitives::IbvCq;
use super::primitives::IbvQp;
use super::primitives::IbvQpInfo;
use super::primitives::IbvWc;
use super::queue_pair::IbvQueuePair;
use super::queue_pair::PollCompletionError;
use super::queue_pair::PollTarget;
use super::queue_pair::WorkRequestError;

/// Queue key for EFA SRD traffic. Both peers must present the same value or the
/// responder drops the traffic silently; [`IbvQpInfo`] carries no queue key, so
/// this is a shared constant rather than something negotiated at connect time.
const EFA_QKEY: u32 = 0x4242;

/// The RDMA operations an EFA SRD queue pair posts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EfaOp {
    Write,
    Read,
}

/// One work request within a posting session: the byte offset into both the
/// local and the remote buffer, the length, and the id its completion carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Chunk {
    offset: usize,
    len: usize,
    wr_id: u64,
}

/// Splits a `total_size`-byte transfer into `chunk_max`-bound work requests,
/// numbering them from `first_wr_id`. A zero-byte transfer still yields one
/// work request, so every posted operation produces a completion.
fn chunks(total_size: usize, chunk_max: usize, first_wr_id: u64) -> Vec<Chunk> {
    assert!(chunk_max > 0, "chunk size must be positive");
    let mut chunks = Vec::with_capacity(total_size.div_ceil(chunk_max).max(1));
    let mut offset = 0;
    let mut remaining = total_size;
    loop {
        let len = std::cmp::min(remaining, chunk_max);
        let wr_id = first_wr_id + chunks.len() as u64;
        chunks.push(Chunk { offset, len, wr_id });
        offset += len;
        remaining -= len;
        if remaining == 0 {
            break;
        }
    }
    chunks
}

/// The address-handle attributes describing a peer reachable at `dgid`, sent
/// from `port_num` using local GID-table entry `sgid_index`.
///
/// Only the destination GID reaches the EFA device: it resolves the handle from
/// `grh.dgid` alone and returns an opaque routing token. The remaining GRH
/// fields — `hop_limit`, `traffic_class`, `sl` — carry IP header values that
/// matter on RoCE v2 and never reach the wire here, so they stay zero.
/// `sgid_index` must still name a populated GID-table entry, because the kernel
/// validates it whenever the handle is global.
fn ah_attr(port_num: u8, dgid: Gid, sgid_index: u8) -> rdmaxcel_sys::ibv_ah_attr {
    rdmaxcel_sys::ibv_ah_attr {
        port_num,
        is_global: 1,
        grh: rdmaxcel_sys::ibv_global_route {
            dgid: rdmaxcel_sys::ibv_gid::from(dgid),
            sgid_index,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// One RDMA work request to add to a [`WrSession`]. Addresses and lengths are
/// resolved by the caller, so building it cannot fail.
#[derive(Debug, Clone, Copy)]
struct Wr {
    op: EfaOp,
    wr_id: u64,
    laddr: u64,
    lkey: u32,
    raddr: u64,
    rkey: u32,
    len: u32,
}

/// An open work-request session on one queue pair's extended builder, borrowing
/// both that queue pair and the peer it is routed to for as long as the session
/// lives.
///
/// `ibv_wr_start` takes the send queue's lock, which only `ibv_wr_complete` or
/// `ibv_wr_abort` releases. This guard makes that pairing unconditional: the
/// session opens on construction and, if the guard is dropped without
/// [`Self::post`] — an unwind out of the caller, say — `Drop` aborts it. Leaving
/// it open would strand the lock and deadlock every later post on the queue
/// pair, and `ibv_destroy_qp` would then destroy a held lock.
struct WrSession<'a> {
    qpex: *mut rdmaxcel_sys::ibv_qp_ex,
    rdma_write: unsafe extern "C" fn(*mut rdmaxcel_sys::ibv_qp_ex, u32, u64),
    rdma_read: unsafe extern "C" fn(*mut rdmaxcel_sys::ibv_qp_ex, u32, u64),
    set_sge: unsafe extern "C" fn(*mut rdmaxcel_sys::ibv_qp_ex, u32, u64, u32),
    set_ud_addr:
        unsafe extern "C" fn(*mut rdmaxcel_sys::ibv_qp_ex, *mut rdmaxcel_sys::ibv_ah, u32, u32),
    complete: unsafe extern "C" fn(*mut rdmaxcel_sys::ibv_qp_ex) -> std::os::raw::c_int,
    abort: unsafe extern "C" fn(*mut rdmaxcel_sys::ibv_qp_ex),
    /// The queue pair `qpex` points into. Never read: it is held so the borrow
    /// checker keeps the queue pair alive for as long as the builder pointer
    /// derived from it.
    _qp: &'a IbvQp,
    /// The destination every request in this session is routed to. Borrowed, so
    /// the address handle cannot be destroyed while the session is open.
    peer: &'a EfaPeer,
    /// Set once `wr_complete` has run. It releases the lock even when it reports
    /// failure, so past that point [`Drop`] must not abort.
    completed: bool,
}

impl<'a> WrSession<'a> {
    /// Resolves `qp`'s extended work-request builder and opens a session on it,
    /// routed to `peer`.
    ///
    /// Every entry point is resolved before the session opens, so a queue pair
    /// missing one fails here rather than partway through building requests.
    ///
    /// # Safety
    ///
    /// `qp` must hold a non-null, live `ibv_qp`, and `peer`'s address handle must
    /// be a live handle created on the same protection domain as `qp`. Borrowing
    /// `peer` keeps that handle from being destroyed for the life of the session,
    /// but says nothing about which device or domain it addresses.
    unsafe fn start(qp: &'a IbvQp, peer: &'a EfaPeer) -> Result<Self, anyhow::Error> {
        // SAFETY: `qp` holds a live `ibv_qp` (caller contract).
        // `ibv_qp_to_qp_ex` returns null unless the QP was created with
        // `IBV_QP_INIT_ATTR_SEND_OPS_FLAGS`.
        let qpex = unsafe { rdmaxcel_sys::ibv_qp_to_qp_ex(qp.as_ptr()) };
        if qpex.is_null() {
            anyhow::bail!(
                "queue pair has no extended work-request builder; it was not created with IBV_QP_INIT_ATTR_SEND_OPS_FLAGS"
            );
        }

        // SAFETY: `qpex` is non-null (checked above) and points into the live
        // `qp`, so reading its function-pointer fields is sound.
        let ops = unsafe { &*qpex };
        let session = Self {
            qpex,
            rdma_write: ops.wr_rdma_write.ok_or_else(|| missing("wr_rdma_write"))?,
            rdma_read: ops.wr_rdma_read.ok_or_else(|| missing("wr_rdma_read"))?,
            set_sge: ops.wr_set_sge.ok_or_else(|| missing("wr_set_sge"))?,
            set_ud_addr: ops
                .wr_set_ud_addr
                .ok_or_else(|| missing("wr_set_ud_addr"))?,
            complete: ops.wr_complete.ok_or_else(|| missing("wr_complete"))?,
            abort: ops.wr_abort.ok_or_else(|| missing("wr_abort"))?,
            _qp: qp,
            peer,
            completed: false,
        };
        let start = ops.wr_start.ok_or_else(|| missing("wr_start"))?;
        // SAFETY: `qpex` is the live builder resolved above. From here the
        // send-queue lock is held, and `session`'s `Drop` releases it on any
        // path that does not reach `post`.
        unsafe { start(qpex) };
        Ok(session)
    }

    /// Adds one signaled work request to the session.
    fn add(&mut self, wr: Wr) {
        // SAFETY: `self.qpex` is the live builder this open session was started
        // on, and `self.peer`'s address handle is borrowed for the session, so it
        // is still live. Each request is fully specified — `wr_id`/`wr_flags`,
        // then the opcode builder, then the scatter/gather entry, then the
        // destination — before the next one begins.
        unsafe {
            // The builder call below reads both of these, so they are set per
            // request rather than once per session.
            (*self.qpex).wr_id = wr.wr_id;
            (*self.qpex).wr_flags = rdmaxcel_sys::ibv_send_flags::IBV_SEND_SIGNALED.0;
            match wr.op {
                EfaOp::Write => (self.rdma_write)(self.qpex, wr.rkey, wr.raddr),
                EfaOp::Read => (self.rdma_read)(self.qpex, wr.rkey, wr.raddr),
            }
            // Order matters: `wr_set_sge` dispatches on the opcode the builder
            // just wrote.
            (self.set_sge)(self.qpex, wr.lkey, wr.laddr, wr.len);
            (self.set_ud_addr)(
                self.qpex,
                self.peer.ah.as_ptr(),
                self.peer.remote_qpn,
                self.peer.qkey,
            );
        }
    }

    /// Closes the session, posting every request added to it.
    ///
    /// All-or-nothing: on failure the provider rolls the whole session back, so
    /// no completion arrives for any of its requests.
    fn post(mut self) -> Result<(), anyhow::Error> {
        // SAFETY: `self.qpex` is the live builder this open session was started
        // on.
        let errno = unsafe { (self.complete)(self.qpex) };
        // `wr_complete` releases the send-queue lock whether or not it
        // succeeded, so the session is closed either way and `Drop` must not
        // abort it. Nothing may be inserted above this line: `Drop` would then
        // be reachable for a session that is already unlocked.
        self.completed = true;
        if errno != 0 {
            return Err(anyhow::anyhow!(
                "failed to post work-request session: {}",
                Error::from_raw_os_error(errno)
            ));
        }
        Ok(())
    }
}

impl Drop for WrSession<'_> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        // Abandoned without posting — an unwind out of the caller, say. Roll the
        // session back so the send-queue lock is released; leaving it held would
        // deadlock every later post on this queue pair.
        // SAFETY: the session is still open (`completed` is false) and
        // `self.qpex` is the live builder it was started on.
        unsafe { (self.abort)(self.qpex) };
    }
}

fn missing(verb: &str) -> anyhow::Error {
    anyhow::anyhow!("EFA queue pair is missing the {} extended verb", verb)
}

/// Transitions `qp` to the state in `attr`, reporting a failure using
/// the string in `target`.
///
/// # Safety
///
/// `qp` must be a live `ibv_qp` (non-null).
unsafe fn modify_qp(
    qp: *mut rdmaxcel_sys::ibv_qp,
    attr: &mut rdmaxcel_sys::ibv_qp_attr,
    mask: rdmaxcel_sys::ibv_qp_attr_mask,
    target: &str,
) -> Result<(), anyhow::Error> {
    // SAFETY: `qp` is a live `ibv_qp` (caller contract); `attr` is a valid
    // `ibv_qp_attr` whose populated fields match `mask`. `ibv_modify_qp` returns
    // the errno.
    let errno = unsafe { rdmaxcel_sys::ibv_modify_qp(qp, attr, mask.0 as i32) };
    if errno != 0 {
        return Err(anyhow::anyhow!(
            "failed to transition EFA queue pair to {}: {}",
            target,
            Error::from_raw_os_error(errno)
        ));
    }
    Ok(())
}

/// Queries the EFA-specific attributes of the device behind `context`.
///
/// # Safety
///
/// `context` must be a live `ibv_context` belonging to an EFA device.
unsafe fn query_device(
    context: *mut rdmaxcel_sys::ibv_context,
) -> Result<rdmaxcel_sys::efadv_device_attr, anyhow::Error> {
    let mut attr = rdmaxcel_sys::efadv_device_attr::default();
    // SAFETY: `context` is a live EFA device context (caller contract); the
    // out-param is a writable, properly aligned `efadv_device_attr` whose size
    // we pass as `inlen`. `efadv_query_device` returns the errno.
    let errno = unsafe {
        rdmaxcel_sys::efadv_query_device(
            context,
            &mut attr,
            std::mem::size_of::<rdmaxcel_sys::efadv_device_attr>() as u32,
        )
    };
    if errno != 0 {
        return Err(anyhow::anyhow!(
            "failed to query EFA device attributes: {}",
            Error::from_raw_os_error(errno)
        ));
    }
    Ok(attr)
}

/// The remote endpoint of a connected [`EfaQueuePair`].
///
/// EFA SRD is UD-addressed: the destination is not a queue-pair attribute, so
/// every work request names it explicitly through an address handle, the peer's
/// queue-pair number, and the queue key.
#[derive(Debug)]
struct EfaPeer {
    ah: IbvAh,
    remote_qpn: u32,
    qkey: u32,
}

/// An EFA SRD queue pair, created through `efadv_create_qp_ex` and driven by the
/// extended work-request builder reached via `ibv_qp_to_qp_ex`.
///
/// SRD is a driver queue-pair type that the device runs through the unreliable
/// datagram state machine, so this shares almost nothing with the
/// reliable-connected path: the connection handshake carries a queue key rather
/// than access flags, and each work request supplies its own destination. Only
/// endpoint discovery, state queries, and completion polling are common, and
/// those delegate to the shared helpers in [`super::queue_pair`].
///
/// Single-owner: it owns the [`IbvQp`] — which in turn owns its two completion
/// queues and the protection domain — and destroys them on drop, so the type is
/// intentionally `!Clone`.
#[derive(Debug)]
pub struct EfaQueuePair {
    qp: IbvQp,
    /// Declared after `qp` so the queue pair is destroyed first. Every work
    /// request embeds the address handle's device token, so the handle has to
    /// outlive any request still referencing it.
    peer: Option<EfaPeer>,
    config: IbvConfig,
    /// The source GID, always table entry 0: EFA is not RoCE, so its
    /// `gid_attrs/types` never reports "RoCE v2" and its table holds one entry.
    gid: Gid,
    /// Largest transfer issued as a single work request, from the device's
    /// reported `max_rdma_size`.
    max_msg_size: usize,
    /// Monotonic work-request id, handed out one per posted WR. The extended
    /// verbs carry no internal counter, so the queue pair tracks its own.
    next_wr_id: u64,
}

impl EfaQueuePair {
    /// Posts `op` over `total_size` bytes from `laddr` to `raddr` as a single
    /// work-request session, split into [`Self::max_msg_size`]-bound chunks, and
    /// returns one work-request id per chunk.
    ///
    /// The session is all-or-nothing: on failure `ibv_wr_complete` rolls every
    /// request in it back, so an `Err` means nothing was posted and no
    /// completion will arrive for any of the ids.
    fn post_chunked(
        &mut self,
        op: EfaOp,
        laddr: usize,
        lkey: u32,
        raddr: usize,
        rkey: u32,
        total_size: usize,
    ) -> Result<Vec<u64>, anyhow::Error> {
        if self.peer.is_none() {
            anyhow::bail!("cannot post on an EfaQueuePair that has not been connected");
        }

        // Resolve every request before opening the session, so the window where
        // the send-queue lock is held does no arithmetic and no allocation.
        // `WrSession` makes an unwind there recoverable.
        let plan = chunks(total_size, self.max_msg_size, self.next_wr_id);
        self.next_wr_id += plan.len() as u64;
        let wrs: Vec<Wr> = plan
            .iter()
            .map(|chunk| Wr {
                op,
                wr_id: chunk.wr_id,
                laddr: (laddr + chunk.offset) as u64,
                lkey,
                raddr: (raddr + chunk.offset) as u64,
                rkey,
                len: chunk.len as u32,
            })
            .collect();

        let peer = self.peer.as_ref().expect("checked above");
        // SAFETY: `self.qp` holds the live queue pair created in `new`, and
        // `peer`'s address handle is owned by `self` and outlives the session.
        let mut session = unsafe { WrSession::start(&self.qp, peer) }?;
        for wr in wrs {
            session.add(wr);
        }
        session.post().map_err(|e| {
            anyhow::anyhow!("{:?} session of {} work request(s): {e}", op, plan.len())
        })?;
        Ok(plan.into_iter().map(|chunk| chunk.wr_id).collect())
    }
}

impl IbvQueuePair for EfaQueuePair {
    unsafe fn new<I: IbvDomainImpl<QueuePair = Self>>(
        domain: &IbvDomain<I>,
        config: IbvConfig,
        send_cq: Arc<IbvCq>,
        recv_cq: Arc<IbvCq>,
    ) -> Result<Self, anyhow::Error> {
        tracing::debug!("creating an EfaQueuePair from config {}", config);
        // `IbvDomain`'s `pd` accessor permits null (e.g. a test domain); a real
        // queue pair needs one, so reject null up front. Everything below then
        // has a live context too, since a PD is only ever allocated against one.
        let pd = domain.as_ptr();
        if pd.is_null() {
            anyhow::bail!("cannot create an EfaQueuePair on a null protection domain");
        }
        let context = domain.context().as_ptr();

        // Resolve the source GID up front (before allocating any FFI resources),
        // so a port without a usable GID fails cleanly here.
        let gid = domain.device_info().gid_at(config.port_num, 0)?;

        // SAFETY: `context` is the live context the PD above was allocated
        // against.
        let device_attr = unsafe { query_device(context) }?;
        let required = rdmaxcel_sys::EFADV_DEVICE_ATTR_CAPS_RDMA_READ
            | rdmaxcel_sys::EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE;
        anyhow::ensure!(
            device_attr.device_caps & required == required,
            "EFA device does not support both RDMA read and RDMA write (device_caps: {:#x})",
            device_attr.device_caps
        );
        let max_msg_size = device_attr.max_rdma_size as usize;
        anyhow::ensure!(
            max_msg_size > 0,
            "EFA device reports a maximum RDMA transfer size of 0"
        );

        // EFA accepts exactly one scatter/gather entry per RDMA work request.
        // `EfaDevice::apply_config_defaults` already caps these, but the manager
        // seeds those defaults only when it spawns without an explicit config,
        // so enforce it here rather than trust the caller.
        let mut config = config;
        config.max_send_sge = 1;
        config.max_recv_sge = 1;
        // The queue depths must fit the device. Reject rather than clamp: the
        // owning `QueuePairActor` budgets send-queue credits against the depth in
        // its own config, so quietly granting fewer would let it over-commit.
        anyhow::ensure!(
            config.max_send_wr <= device_attr.max_sq_wr,
            "configured max_send_wr ({}) exceeds the EFA device's limit ({})",
            config.max_send_wr,
            device_attr.max_sq_wr
        );
        anyhow::ensure!(
            config.max_recv_wr <= device_attr.max_rq_wr,
            "configured max_recv_wr ({}) exceeds the EFA device's limit ({})",
            config.max_recv_wr,
            device_attr.max_rq_wr
        );

        // An SRD queue pair: a driver queue-pair type selected through
        // `efadv_qp_init_attr`. The send-ops flags both request the RDMA
        // builders and are what makes `ibv_qp_to_qp_ex` yield a builder at all.
        let mut init_attr = rdmaxcel_sys::ibv_qp_init_attr_ex {
            send_cq: send_cq.as_ptr(),
            recv_cq: recv_cq.as_ptr(),
            cap: rdmaxcel_sys::ibv_qp_cap {
                max_send_wr: config.max_send_wr,
                max_recv_wr: config.max_recv_wr,
                max_send_sge: config.max_send_sge,
                max_recv_sge: config.max_recv_sge,
                max_inline_data: 0,
            },
            qp_type: rdmaxcel_sys::ibv_qp_type::IBV_QPT_DRIVER,
            sq_sig_all: 0,
            pd,
            comp_mask: rdmaxcel_sys::IBV_QP_INIT_ATTR_PD
                | rdmaxcel_sys::IBV_QP_INIT_ATTR_SEND_OPS_FLAGS,
            send_ops_flags: (rdmaxcel_sys::IBV_QP_EX_WITH_RDMA_WRITE
                | rdmaxcel_sys::IBV_QP_EX_WITH_RDMA_READ) as u64,
            ..Default::default()
        };
        let mut efa_attr = rdmaxcel_sys::efadv_qp_init_attr {
            driver_qp_type: rdmaxcel_sys::EFADV_QP_DRIVER_TYPE_SRD,
            ..Default::default()
        };
        // `efadv_create_qp_ex` writes the granted queue depths back into
        // `init_attr.cap`, hence the mutable borrow.
        // SAFETY: `context` and `pd` are non-null (checked above) and live; both
        // attr structs are fully initialized and outlive the call, and
        // `init_attr`'s CQ pointers came from the freshly created
        // `send_cq`/`recv_cq`. `efadv_create_qp_ex` returns null on failure.
        let qp = unsafe {
            rdmaxcel_sys::efadv_create_qp_ex(
                context,
                &mut init_attr,
                &mut efa_attr,
                std::mem::size_of::<rdmaxcel_sys::efadv_qp_init_attr>() as u32,
            )
        };
        if qp.is_null() {
            anyhow::bail!(
                "failed to create EFA SRD queue pair (QP): {}",
                Error::last_os_error()
            );
        }

        // SAFETY: `qp` is a live SRD QP just created against `pd` with
        // `send_cq`/`recv_cq`; `IbvQp` holds a clone of each, keeping them alive
        // for at least as long as the QP it destroys on drop.
        let qp = unsafe { IbvQp::from_raw(qp, send_cq, recv_cq, domain.pd().clone()) };

        Ok(Self {
            qp,
            peer: None,
            config,
            gid,
            max_msg_size,
            next_wr_id: 0,
        })
    }

    fn connect(&mut self, info: &IbvQpInfo) -> Result<(), anyhow::Error> {
        let Some(dgid) = info.gid else {
            anyhow::bail!(
                "EFA addresses peers by GID, but the peer endpoint {:?} carries none",
                info
            );
        };

        // Build the address handle before the transitions. It is a
        // protection-domain operation that does not depend on queue-pair state,
        // so failing here leaves the queue pair in RESET rather than in RTS with
        // no route — a state in which every post would fail.
        let mut attr = ah_attr(self.config.port_num, dgid, self.gid.index());
        // SAFETY: the queue pair was created against this PD, which it keeps
        // alive; `attr` is fully initialized and outlives the call.
        let ah = unsafe { IbvAh::create(self.qp.pd().clone(), &mut attr) }?;

        // The device runs an SRD queue pair through the unreliable datagram
        // state machine: INIT carries the queue key in place of access flags,
        // RTR carries nothing but the state, and RTS only the send-queue packet
        // sequence number. None of the reliable-connected path attributes apply,
        // because the destination travels with each work request instead.
        let qp = self.qp.as_ptr();
        let mut attr = rdmaxcel_sys::ibv_qp_attr {
            qp_state: rdmaxcel_sys::ibv_qp_state::IBV_QPS_INIT,
            qkey: EFA_QKEY,
            pkey_index: self.config.pkey_index,
            port_num: self.config.port_num,
            ..Default::default()
        };
        let mask = rdmaxcel_sys::ibv_qp_attr_mask::IBV_QP_STATE
            | rdmaxcel_sys::ibv_qp_attr_mask::IBV_QP_PKEY_INDEX
            | rdmaxcel_sys::ibv_qp_attr_mask::IBV_QP_PORT
            | rdmaxcel_sys::ibv_qp_attr_mask::IBV_QP_QKEY;
        // SAFETY: `qp` is the live queue pair, kept alive for `self`'s lifetime.
        unsafe { modify_qp(qp, &mut attr, mask, "INIT") }?;

        let mut attr = rdmaxcel_sys::ibv_qp_attr {
            qp_state: rdmaxcel_sys::ibv_qp_state::IBV_QPS_RTR,
            ..Default::default()
        };
        let mask = rdmaxcel_sys::ibv_qp_attr_mask::IBV_QP_STATE;
        // SAFETY: as for the INIT transition above.
        unsafe { modify_qp(qp, &mut attr, mask, "RTR") }?;

        let mut attr = rdmaxcel_sys::ibv_qp_attr {
            qp_state: rdmaxcel_sys::ibv_qp_state::IBV_QPS_RTS,
            sq_psn: self.config.psn,
            ..Default::default()
        };
        let mask = rdmaxcel_sys::ibv_qp_attr_mask::IBV_QP_STATE
            | rdmaxcel_sys::ibv_qp_attr_mask::IBV_QP_SQ_PSN;
        // SAFETY: as for the INIT transition above.
        unsafe { modify_qp(qp, &mut attr, mask, "RTS") }?;

        self.peer = Some(EfaPeer {
            ah,
            remote_qpn: info.qp_num,
            qkey: EFA_QKEY,
        });
        tracing::debug!(
            "EfaQueuePair reached RTS and is routed to {:?} (qp: {:?})",
            info,
            qp
        );
        Ok(())
    }

    fn get_qp_info(&mut self) -> Result<IbvQpInfo, anyhow::Error> {
        let context = self.qp.context().as_ptr();
        // SAFETY: `self.qp` is the live queue pair and `context` its non-null
        // device context (both validated in `new`), valid for `self`'s lifetime.
        unsafe { super::queue_pair::get_qp_info(self.qp.as_ptr(), context, &self.config, self.gid) }
    }

    fn state(&mut self) -> Result<u32, anyhow::Error> {
        // SAFETY: `self.qp` is the live queue pair, kept alive for `self`'s
        // lifetime.
        unsafe { super::queue_pair::state(self.qp.as_ptr()) }
    }

    fn max_msg_size(&self) -> usize {
        self.max_msg_size
    }

    fn put(
        &mut self,
        remote_dst: IbvRemoteMemoryRegionView,
        local_src: IbvMemoryRegionView,
    ) -> Result<Vec<u64>, anyhow::Error> {
        if remote_dst.size < local_src.size {
            return Err(anyhow::anyhow!(
                "remote buffer size ({}) is smaller than local buffer size ({})",
                remote_dst.size,
                local_src.size
            ));
        }
        self.post_chunked(
            EfaOp::Write,
            local_src.rdma_addr,
            local_src.lkey,
            remote_dst.addr,
            remote_dst.rkey,
            local_src.size,
        )
    }

    fn get(
        &mut self,
        local_dst: IbvMemoryRegionView,
        remote_src: IbvRemoteMemoryRegionView,
    ) -> Result<Vec<u64>, anyhow::Error> {
        if local_dst.size < remote_src.size {
            return Err(anyhow::anyhow!(
                "local buffer size ({}) is smaller than remote buffer size ({})",
                local_dst.size,
                remote_src.size
            ));
        }
        self.post_chunked(
            EfaOp::Read,
            local_dst.rdma_addr,
            local_dst.lkey,
            remote_src.addr,
            remote_src.rkey,
            remote_src.size,
        )
    }

    fn poll_completion(
        &mut self,
        target: PollTarget,
    ) -> Result<Option<Result<IbvWc, WorkRequestError>>, PollCompletionError> {
        // SAFETY: `self.qp` owns the live queue pair built in `new`, along with
        // its completion queues and device context, all non-null and alive for
        // `self`'s lifetime. `&mut self` excludes another poll through this queue
        // pair, and its lease leaves it the only queue pair polling that
        // completion queue, so no other thread is polling it.
        unsafe { super::queue_pair::poll_one(&self.qp, target) }
    }
}

#[cfg(test)]
mod tests {
    use std::net::Ipv6Addr;
    use std::time::Duration;
    use std::time::Instant;

    use super::*;
    use crate::backend::ibverbs::device::IbvDevice;
    use crate::backend::ibverbs::device::IbvDeviceImpl;
    use crate::backend::ibverbs::efa_device::EfaDevice;
    use crate::backend::ibverbs::efa_domain::EfaDomain;
    use crate::local_memory::KeepaliveLocalMemory;

    // A transfer that fits in one work request still gets exactly one, and it
    // is numbered from the id handed in.
    #[test]
    fn chunks_fitting_transfer_yields_one_wr() {
        assert_eq!(
            chunks(1024, 4096, 7),
            vec![Chunk {
                offset: 0,
                len: 1024,
                wr_id: 7
            }]
        );
    }

    // A zero-byte transfer still yields one work request, so the caller always
    // receives a completion to match against.
    #[test]
    fn chunks_zero_byte_transfer_yields_one_wr() {
        assert_eq!(
            chunks(0, 4096, 0),
            vec![Chunk {
                offset: 0,
                len: 0,
                wr_id: 0
            }]
        );
    }

    // An oversized transfer splits into consecutive, contiguous chunks with a
    // short final one, each carrying the next id.
    #[test]
    fn chunks_oversized_transfer_splits_with_short_tail() {
        let plan = chunks(2500, 1000, 100);
        assert_eq!(
            plan,
            vec![
                Chunk {
                    offset: 0,
                    len: 1000,
                    wr_id: 100
                },
                Chunk {
                    offset: 1000,
                    len: 1000,
                    wr_id: 101
                },
                Chunk {
                    offset: 2000,
                    len: 500,
                    wr_id: 102
                },
            ]
        );
        let total: usize = plan.iter().map(|chunk| chunk.len).sum();
        assert_eq!(total, 2500, "chunks must cover the transfer exactly");
    }

    // An exact multiple of the chunk size produces no trailing empty request.
    #[test]
    fn chunks_exact_multiple_has_no_empty_tail() {
        let plan = chunks(2000, 1000, 0);
        assert_eq!(plan.len(), 2, "2000 bytes at 1000 per WR is two requests");
        assert!(plan.iter().all(|chunk| chunk.len == 1000));
    }

    // The address handle carries the peer's GID and the *local* source-GID
    // index, and leaves the RoCE-only GRH fields zero.
    #[test]
    fn ah_attr_carries_peer_gid_and_local_sgid_index() {
        let peer = Gid::for_test(Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 2), 3);
        let attr = ah_attr(1, peer, 0);

        assert_eq!(attr.port_num, 1);
        assert_eq!(attr.is_global, 1, "EFA requires a global address handle");
        // SAFETY: `raw` is one arm of the `ibv_gid` union and is always
        // initialized; reading the 16 address bytes back is sound.
        let dgid = unsafe { attr.grh.dgid.raw };
        assert_eq!(
            dgid,
            Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 2).octets(),
            "the handle must address the peer's GID"
        );
        assert_eq!(
            attr.grh.sgid_index, 0,
            "sgid_index names the local GID table entry, not the peer's index 3"
        );
        assert_eq!(attr.grh.hop_limit, 0, "hop_limit is RoCE-only");
        assert_eq!(attr.grh.traffic_class, 0, "traffic_class is RoCE-only");
        assert_eq!(attr.sl, 0);
        assert_eq!(attr.dlid, 0, "EFA has no LID");
    }

    // =====================================================================
    // Shared scaffolding for the EFA hardware probes below.
    // =====================================================================
    //
    // The three probes (same-device two-QP, single-QP loopback, and
    // cross-node) all open an EFA device, register host buffers, drive SRD
    // WRITEs, and drain send completions. These helpers hold that common
    // machinery so each probe reads as just its own experiment.

    /// The probe transfer size, in bytes.
    const PROBE_LEN: usize = 4096;

    /// A recognizable, position-dependent byte for offset `i`, so a WRITE that
    /// lands can be verified against the source it came from.
    fn probe_pattern(i: usize) -> u8 {
        (i as u8).wrapping_mul(31).wrapping_add(7)
    }

    /// The outcome of draining one send-queue completion.
    #[derive(Debug)]
    enum Outcome {
        /// A success completion (`IBV_WC_SUCCESS`) for `wr_id`.
        Success(u64),
        /// A per-WR failure completion.
        Failed(WorkRequestError),
        /// No completion arrived before the deadline.
        TimedOut,
        /// `ibv_poll_cq` itself failed; the CQ is unusable.
        PollBroke(String),
    }

    impl Outcome {
        fn is_success(&self) -> bool {
            matches!(self, Outcome::Success(_))
        }
        fn is_failure(&self) -> bool {
            matches!(self, Outcome::Failed(_))
        }
    }

    /// A human-readable name for the `ibv_wc_status` values these probes can
    /// observe, so a printed verdict is legible without a lookup table.
    fn wc_status_name(status: rdmaxcel_sys::ibv_wc_status::Type) -> &'static str {
        use rdmaxcel_sys::ibv_wc_status::*;
        match status {
            IBV_WC_SUCCESS => "IBV_WC_SUCCESS",
            IBV_WC_LOC_LEN_ERR => "IBV_WC_LOC_LEN_ERR",
            IBV_WC_LOC_QP_OP_ERR => "IBV_WC_LOC_QP_OP_ERR",
            IBV_WC_LOC_PROT_ERR => "IBV_WC_LOC_PROT_ERR",
            IBV_WC_WR_FLUSH_ERR => "IBV_WC_WR_FLUSH_ERR",
            IBV_WC_REM_INV_REQ_ERR => "IBV_WC_REM_INV_REQ_ERR",
            IBV_WC_REM_ACCESS_ERR => "IBV_WC_REM_ACCESS_ERR",
            IBV_WC_REM_OP_ERR => "IBV_WC_REM_OP_ERR",
            IBV_WC_RNR_RETRY_EXC_ERR => "IBV_WC_RNR_RETRY_EXC_ERR",
            IBV_WC_REM_ABORT_ERR => "IBV_WC_REM_ABORT_ERR",
            IBV_WC_REM_INV_RD_REQ_ERR => "IBV_WC_REM_INV_RD_REQ_ERR",
            IBV_WC_BAD_RESP_ERR => "IBV_WC_BAD_RESP_ERR",
            _ => "IBV_WC_<other>",
        }
    }

    fn describe(outcome: &Outcome) -> String {
        match outcome {
            Outcome::Success(id) => format!("SUCCESS (wr_id={id}) -- no remote error reported"),
            Outcome::Failed(err) => format!(
                "FAILED {} (raw={:?}, vendor_err={}, wr_id={}) -- responder rejected the access",
                wc_status_name(err.status),
                err.status,
                err.vendor_err,
                err.wr_id,
            ),
            Outcome::TimedOut => "TIMED OUT (no completion before the deadline)".to_string(),
            Outcome::PollBroke(msg) => format!("CQ POLL FAILED: {msg}"),
        }
    }

    /// Drain exactly one send-queue completion, waiting up to `timeout`.
    fn drain_one(qp: &mut EfaQueuePair, timeout: Duration) -> Outcome {
        let start = Instant::now();
        loop {
            match qp.poll_completion(PollTarget::Send) {
                Ok(Some(Ok(wc))) => return Outcome::Success(wc.wr_id()),
                Ok(Some(Err(err))) => return Outcome::Failed(err),
                Ok(None) => {}
                Err(err) => return Outcome::PollBroke(err.to_string()),
            }
            if start.elapsed() >= timeout {
                return Outcome::TimedOut;
            }
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    /// Post one RDMA WRITE (`put`) and drain its single send completion.
    fn write_and_drain(
        qp: &mut EfaQueuePair,
        dst: IbvRemoteMemoryRegionView,
        src: IbvMemoryRegionView,
        timeout: Duration,
    ) -> Outcome {
        qp.put(dst, src)
            .expect("posting a WRITE should succeed locally");
        drain_one(qp, timeout)
    }

    /// Modest SRD config for these probes: small queue depths (the device caps
    /// are far higher, and `EfaQueuePair::new` rejects a config that exceeds
    /// them) over host memory (no GPUDirect).
    fn probe_config() -> IbvConfig {
        let mut config = IbvConfig::default();
        EfaDevice::apply_config_defaults(&mut config);
        config.max_send_wr = 16;
        config.max_recv_wr = 16;
        config.use_gpu_direct = false;
        config
    }

    /// The name of the first visible EFA device, or `None` when this host lacks
    /// the verbs stack -- letting a probe self-skip off EFA hardware.
    fn first_efa_nic() -> Option<String> {
        IbvDevice::<EfaDevice>::list()
            .first()
            .map(|nic| nic.name().clone())
    }

    /// Create one SRD queue pair on `domain`, with its own modest CQ.
    fn make_srd_qp(domain: &IbvDomain<EfaDomain>, config: &IbvConfig) -> EfaQueuePair {
        let cq =
            Arc::new(unsafe { IbvCq::create(domain.context().clone(), 64) }.expect("create CQ"));
        domain.create_queue_pair(config, cq).expect("create SRD QP")
    }

    /// Register a source MR filled with the recognizable probe pattern.
    ///
    /// The returned `KeepaliveLocalMemory` must be kept alive for as long as the
    /// MR is used: the view keeps the *registration* alive, not the backing
    /// buffer.
    fn register_pattern_src(
        domain: &IbvDomain<EfaDomain>,
    ) -> (KeepaliveLocalMemory, IbvMemoryRegionView) {
        let data: Box<[u8]> = (0..PROBE_LEN)
            .map(probe_pattern)
            .collect::<Vec<u8>>()
            .into_boxed_slice();
        let mem = KeepaliveLocalMemory::try_new(Arc::new(data)).expect("wrap source buffer");
        let view = domain.register_mr(&mem).expect("register source MR");
        (mem, view)
    }

    /// Register a zeroed MR sized to the probe pattern. See
    /// [`register_pattern_src`] on keeping the returned handle alive.
    fn register_zeroed(
        domain: &IbvDomain<EfaDomain>,
    ) -> (KeepaliveLocalMemory, IbvMemoryRegionView) {
        let mem = KeepaliveLocalMemory::try_new(Arc::new(vec![0u8; PROBE_LEN].into_boxed_slice()))
            .expect("wrap zeroed buffer");
        let view = domain.register_mr(&mem).expect("register destination MR");
        (mem, view)
    }

    /// A remote view identical to `good` but with an unregistered rkey -- a
    /// fault only the responder can detect.
    fn bad_rkey_view(good: &IbvRemoteMemoryRegionView) -> IbvRemoteMemoryRegionView {
        let bad_rkey = good.rkey ^ 0x00ff_ffff;
        assert_ne!(bad_rkey, good.rkey, "corrupted rkey must differ");
        IbvRemoteMemoryRegionView {
            rkey: bad_rkey,
            ..good.clone()
        }
    }

    /// A remote view identical to `good` but addressing 1 GiB past the region
    /// -- again detectable only by the responder.
    fn out_of_bounds_view(good: &IbvRemoteMemoryRegionView) -> IbvRemoteMemoryRegionView {
        IbvRemoteMemoryRegionView {
            addr: good.addr + (1usize << 30),
            ..good.clone()
        }
    }

    /// Inject the two responder-only faults against a connected `qp` whose
    /// valid remote view is `good_dst`: a WRITE with an unregistered rkey, then
    /// a WRITE to an address outside the region. Print each outcome under
    /// `label` and return the *decisive* bad-rkey outcome -- only the responder
    /// can detect either fault, so a failure means the initiator waited for it.
    fn probe_remote_faults(
        qp: &mut EfaQueuePair,
        good_dst: &IbvRemoteMemoryRegionView,
        src: &IbvMemoryRegionView,
        timeout: Duration,
        label: &str,
    ) -> Outcome {
        // Decisive probe: WRITE with an unregistered remote rkey.
        let bad_rkey_outcome = write_and_drain(qp, bad_rkey_view(good_dst), src.clone(), timeout);
        println!("{label}[2] bad-rkey WRITE: {}", describe(&bad_rkey_outcome));

        // Corroboration: valid rkey, remote address far outside the region.
        let oob_outcome = write_and_drain(qp, out_of_bounds_view(good_dst), src.clone(), timeout);
        println!("{label}[3] oob-addr WRITE: {}", describe(&oob_outcome));

        bad_rkey_outcome
    }

    /// Render the probe verdict from the decisive bad-rkey `outcome` and assert
    /// it was a *failure* completion -- i.e. the send completion is NOT
    /// local-only. Kept separate from [`probe_remote_faults`] so a caller can
    /// release a peer between posting the faults and this (possibly panicking)
    /// assertion.
    fn assert_not_local_only(outcome: &Outcome) {
        let succeeded = outcome.is_success();
        let failed = outcome.is_failure();

        println!("\n==================== VERDICT ====================");
        if failed {
            println!(
                "EFA SRD send-queue completions are NOT local-only.\n\
                 A WRITE whose only fault is on the responder surfaced a failure\n\
                 completion on the initiator, so the completion is generated only\n\
                 after the remote NIC acknowledges and validates the access, not\n\
                 when the local NIC accepts the send."
            );
        } else if succeeded {
            println!(
                "EFA SRD send-queue completions appear LOCAL-ONLY.\n\
                 A WRITE with an unregistered remote rkey still completed with\n\
                 success on the initiator."
            );
        } else {
            println!("INCONCLUSIVE (no clear completion for the bad-rkey WRITE).");
        }
        println!("=================================================\n");

        assert!(
            !succeeded,
            "hypothesis check: the bad-rkey WRITE completed successfully, which \
             would mean SRD send completions are local-only -- contradicting \
             rdma-core's REMOTE_ERROR_* completion statuses"
        );
        assert!(
            failed,
            "expected the bad-rkey WRITE to complete with a failure status \
             (IBV_WC_REM_ACCESS_ERR), got {outcome:?}"
        );
    }

    // =====================================================================
    // Hardware probe: are EFA SRD send-queue completions local-only?
    // =====================================================================
    //
    // The question: when an SRD queue pair posts a send-queue work request
    // (here an RDMA WRITE), does a *successful* completion mean only that the
    // local NIC accepted/transmitted the work, or that the remote NIC
    // acknowledged it (and, for a WRITE, that the remote memory access
    // succeeded)?
    //
    // The probe: post WRITEs whose only fault is on the responder and see
    // whether the failure surfaces on the *initiator's* send completion.
    //   1. a control WRITE that is entirely valid (anchors "success");
    //   2. a WRITE with an unregistered remote rkey;
    //   3. a WRITE with a valid rkey but a remote address outside the MR.
    // Neither fault in (2)/(3) is detectable by the local NIC at post time --
    // the rkey/IOVA table lives on the responder -- so:
    //   * if the initiator's completion is SUCCESS  => it is LOCAL-ONLY;
    //   * if the initiator's completion is an ERROR => the completion waited
    //     for the responder's acknowledgement (the local NIC could only learn
    //     the access was rejected from the responder's NAK).
    //
    // This is the empirical counterpart to what rdma-core's efa provider
    // already encodes: `to_ibv_status` (providers/efa/verbs.c) maps the
    // responder-originated `EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS`
    // ("RKEY not registered or does not match remote IOVA") to
    // `IBV_WC_REM_ACCESS_ERR` on the sender's CQE -- a status the sender can
    // only produce after a round trip. So the expected verdict is
    // "NOT local-only", and a bad-rkey WRITE should complete with
    // `IBV_WC_REM_ACCESS_ERR`.
    //
    // Unlike a reliable-connected QP, an SRD QP does *not* enter the error
    // state when a single WR fails, so ops (2) and (3) are independent rather
    // than the second being flushed behind the first.
    //
    // Self-skips when no EFA device is visible (e.g. a host or container with
    // no `/dev/infiniband` + `/sys/class/infiniband` verbs stack). Run on an
    // EFA-provisioned node with:
    //
    //   cargo test --lib -p monarch_rdma efa_srd_send_completion_locality -- --nocapture
    #[test]
    fn efa_srd_send_completion_locality() {
        let Some(nic) = first_efa_nic() else {
            println!(
                "SKIP efa_srd_send_completion_locality: no EFA device visible \
                 (ibv_get_device_list returned none). This host/container lacks \
                 the verbs stack (/dev/infiniband + /sys/class/infiniband); run \
                 on an EFA-provisioned node, e.g. under `srun`."
            );
            return;
        };
        println!("Probing EFA device `{nic}` for SRD send-completion locality\n");

        let config = probe_config();
        let mut device = IbvDevice::<EfaDevice>::try_open(&nic, config.clone())
            .expect("EFA device from IbvDevice::list should open");
        let domain = device
            .get_or_create_domain("srd-locality-probe")
            .expect("EFA domain creation should succeed");

        // `_src_mem`/`dst_mem` keep the registered buffers mapped for the MRs'
        // lifetime (a view keeps its registration, not its backing memory);
        // `dst_mem` is also read back after the control WRITE.
        let (_src_mem, src_view) = register_pattern_src(domain);
        let (dst_mem, dst_view) = register_zeroed(domain);

        // Initiator and responder SRD QPs on the same device, cross-connected:
        // EFA requires the responder to hold a valid address handle back to the
        // initiator for RDMA ops (otherwise the responder reports
        // REMOTE_ERROR_UNKNOWN_PEER), so both sides must connect.
        let mut initiator = make_srd_qp(domain, &config);
        let mut responder = make_srd_qp(domain, &config);
        let init_info = initiator.get_qp_info().expect("initiator QP info");
        let resp_info = responder.get_qp_info().expect("responder QP info");
        initiator
            .connect(&resp_info)
            .expect("connect initiator -> responder");
        responder
            .connect(&init_info)
            .expect("connect responder -> initiator");

        let src = src_view.clone();
        let good_dst = IbvRemoteMemoryRegionView::from(&dst_view);
        let timeout = Duration::from_secs(10);

        // 1. Control: a fully valid WRITE must succeed *and* its bytes must
        //    land at the responder. This ties a success completion to the data
        //    actually being applied remotely.
        let control = write_and_drain(&mut initiator, good_dst.clone(), src.clone(), timeout);
        assert!(
            control.is_success(),
            "control WRITE should complete successfully, got {control:?}"
        );
        let mut landed = vec![0u8; PROBE_LEN];
        // SAFETY: `dst_mem` is the sole handle to this host allocation and no
        // other thread accesses it; the WRITE above has already completed.
        unsafe { dst_mem.read_at(0, &mut landed) }.expect("read destination");
        let want: Vec<u8> = (0..PROBE_LEN).map(probe_pattern).collect();
        assert_eq!(landed, want, "control WRITE did not land at the responder");
        println!("  [1] control WRITE : SUCCESS, {PROBE_LEN} bytes verified at the responder");

        // 2 & 3. Inject the two responder-only faults, then render the verdict.
        let bad_rkey = probe_remote_faults(&mut initiator, &good_dst, &src, timeout, "  ");
        assert_not_local_only(&bad_rkey);

        // Keep the responder alive until every WRITE above has completed: its
        // QP is the destination QP and holds the address handle the responder
        // needs to acknowledge the initiator.
        drop(responder);
    }

    // =====================================================================
    // Single-QP loopback: does a self-connected SRD QP work at all?
    // =====================================================================
    //
    // The tightest possible local configuration: a *single* SRD QP connected
    // to itself, so it is simultaneously the initiator and the responder and
    // every WRITE loops back through one NIC into memory the same QP owns.
    // `efa_srd_send_completion_locality` uses two cross-connected QPs on one
    // device; this collapses that to one self-connected QP.
    //
    // This is only a liveness check -- it proves the loopback path carries data
    // end to end: a valid WRITE completes and its bytes land in the destination
    // buffer. It deliberately does *not* inject the responder-only faults; the
    // send-completion locality question is answered by
    // `efa_srd_send_completion_locality` (same device) and
    // `efa_srd_send_completion_locality_xnode` (across the fabric).
    //
    // Self-skips when no EFA device is visible. Run on an EFA-provisioned node
    // with:
    //
    //   cargo test --lib -p monarch_rdma efa_srd_send_completion_locality_loopback -- --nocapture
    #[test]
    fn efa_srd_send_completion_locality_loopback() {
        let Some(nic) = first_efa_nic() else {
            println!(
                "SKIP efa_srd_send_completion_locality_loopback: no EFA device visible \
                 (ibv_get_device_list returned none). This host/container lacks the \
                 verbs stack (/dev/infiniband + /sys/class/infiniband); run on an \
                 EFA-provisioned node, e.g. under `srun`."
            );
            return;
        };
        println!("Exercising a single-QP loopback WRITE on EFA device `{nic}`\n");

        let config = probe_config();
        let mut device = IbvDevice::<EfaDevice>::try_open(&nic, config.clone())
            .expect("EFA device from IbvDevice::list should open");
        let domain = device
            .get_or_create_domain("srd-loopback")
            .expect("EFA domain creation should succeed");

        // `_src_mem`/`dst_mem` keep the registered buffers mapped for the MRs'
        // lifetime; `dst_mem` is read back to confirm the bytes looped through.
        let (_src_mem, src_view) = register_pattern_src(domain);
        let (dst_mem, dst_view) = register_zeroed(domain);

        // One SRD QP connected to itself: it is both initiator and responder,
        // so a WRITE loops back through this NIC into memory the same QP owns.
        // EFA still requires a valid address handle back to the "peer" for RDMA
        // ops; here the peer is us, so we connect to our own QP info (otherwise
        // the responder side reports REMOTE_ERROR_UNKNOWN_PEER).
        let mut qp = make_srd_qp(domain, &config);
        let my_info = qp.get_qp_info().expect("loopback QP info");
        qp.connect(&my_info).expect("connect loopback QP to itself");

        let src = src_view.clone();
        let good_dst = IbvRemoteMemoryRegionView::from(&dst_view);
        let timeout = Duration::from_secs(10);

        // A single valid WRITE must complete successfully and its bytes must
        // land in the destination buffer -- proving the loopback path carries
        // data end to end through one self-connected QP. (This is only a
        // liveness check; the responder-only fault probes that decide the
        // locality question live in `efa_srd_send_completion_locality`.)
        let control = write_and_drain(&mut qp, good_dst, src, timeout);
        assert!(
            control.is_success(),
            "loopback WRITE should complete successfully, got {control:?}"
        );
        let mut landed = vec![0u8; PROBE_LEN];
        // SAFETY: `dst_mem` is the sole handle to this host allocation and no
        // other thread accesses it; the WRITE above has already completed.
        unsafe { dst_mem.read_at(0, &mut landed) }.expect("read destination");
        let want: Vec<u8> = (0..PROBE_LEN).map(probe_pattern).collect();
        assert_eq!(
            landed, want,
            "loopback WRITE did not land in the destination buffer"
        );
        println!("  loopback WRITE: SUCCESS, {PROBE_LEN} bytes verified via a self-connected QP");
    }

    // =====================================================================
    // Cross-node variant of the locality probe.
    // =====================================================================
    //
    // Same question as `efa_srd_send_completion_locality`, but the initiator
    // and responder run in *separate processes on separate nodes*, so every
    // WRITE crosses the real EFA fabric between two hosts instead of looping
    // back through one NIC. That rules out any same-NIC shortcut: a
    // remote-only fault (bad rkey / out-of-bounds remote address) that still
    // surfaces on the initiator's completion proves the completion waited for
    // the *other host's* NIC to acknowledge (and validate) the access.
    //
    // The fault injection has to live at this `EfaQueuePair` /
    // `IbvRemoteMemoryRegionView` layer -- the high-level Python `RDMABuffer`
    // API never exposes an rkey to
    // corrupt -- so the data path is this Rust test on both nodes. Only the
    // launch is external: run this same test binary as two tasks (one per
    // node), e.g. under `srun -N2 --ntasks-per-node=1`. The two ranks find
    // each other through a shared-filesystem rendezvous directory, over which
    // they exchange QP endpoint info and the destination buffer as JSON.
    //
    // Role comes from `EFA_PROBE_ROLE` (`responder`|`initiator`), or from
    // `SLURM_PROCID` (0 => responder) when that is unset. The rendezvous
    // directory comes from `EFA_PROBE_RENDEZVOUS`. The test self-skips when no
    // role is set, so an ordinary `cargo test` never runs it.
    //
    //   # Build once on the shared filesystem, then run one task per node:
    //   cargo test --lib -p monarch_rdma efa_srd_send_completion_locality_xnode --no-run
    //   # (note the "Executable ... (target/debug/deps/monarch_rdma-<hash>)" path)
    //   RDVZ="$HOME/efa_probe_rdvz.$SLURM_JOB_ID"   # fresh, empty, on shared FS
    //   srun -N2 --ntasks-per-node=1 --gpus-per-node=1 -p <efa-partition> \
    //     env EFA_PROBE_RENDEZVOUS="$RDVZ" \
    //     target/debug/deps/monarch_rdma-<hash> \
    //       --exact backend::ibverbs::efa_queue_pair::tests::efa_srd_send_completion_locality_xnode \
    //       --nocapture --test-threads=1
    #[test]
    fn efa_srd_send_completion_locality_xnode() {
        use std::path::Path;
        use std::path::PathBuf;

        // The responder's advertised endpoint: its QP address plus the
        // destination buffer's keys/address. Both derive serde, so the raw
        // fields (GID bytes included) cross the filesystem without hand
        // encoding anything private.
        #[derive(serde::Serialize, serde::Deserialize)]
        struct Endpoint {
            qp: IbvQpInfo,
            buf: IbvRemoteMemoryRegionView,
        }

        // Atomically publish `value` at `path` (write-then-rename), so a reader
        // on the other node never observes a half-written file.
        fn publish<T: serde::Serialize>(path: &Path, value: &T) {
            let tmp = path.with_extension("tmp");
            std::fs::write(
                &tmp,
                serde_json::to_vec(value).expect("serialize rendezvous payload"),
            )
            .expect("write rendezvous tmp file");
            std::fs::rename(&tmp, path).expect("atomically publish rendezvous file");
        }

        // Poll for a rendezvous file to appear and deserialize it.
        fn await_json<T: serde::de::DeserializeOwned>(path: &Path, timeout: Duration) -> T {
            let start = Instant::now();
            loop {
                if let Ok(bytes) = std::fs::read(path)
                    && !bytes.is_empty()
                    && let Ok(value) = serde_json::from_slice::<T>(&bytes)
                {
                    return value;
                }
                assert!(
                    start.elapsed() < timeout,
                    "timed out after {timeout:?} waiting for {}",
                    path.display()
                );
                std::thread::sleep(Duration::from_millis(50));
            }
        }

        fn await_flag(path: &Path, timeout: Duration) {
            let start = Instant::now();
            while !path.exists() {
                assert!(
                    start.elapsed() < timeout,
                    "timed out after {timeout:?} waiting for flag {}",
                    path.display()
                );
                std::thread::sleep(Duration::from_millis(50));
            }
        }

        #[derive(PartialEq, Eq, Clone, Copy)]
        enum Role {
            Responder,
            Initiator,
        }

        // Role from EFA_PROBE_ROLE, else from SLURM_PROCID (rank 0 = responder).
        let role = match std::env::var("EFA_PROBE_ROLE").ok().as_deref() {
            Some("responder") => Some(Role::Responder),
            Some("initiator") => Some(Role::Initiator),
            Some(other) => {
                panic!("EFA_PROBE_ROLE must be 'responder' or 'initiator', got {other:?}")
            }
            None => std::env::var("SLURM_PROCID")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .map(|procid| {
                    if procid == 0 {
                        Role::Responder
                    } else {
                        Role::Initiator
                    }
                }),
        };
        let Some(role) = role else {
            println!(
                "SKIP efa_srd_send_completion_locality_xnode: set EFA_PROBE_ROLE=responder|initiator \
                 (or launch under srun so SLURM_PROCID is set) plus EFA_PROBE_RENDEZVOUS=<shared dir> \
                 to activate this cross-node probe."
            );
            return;
        };
        let rdvz =
            PathBuf::from(std::env::var("EFA_PROBE_RENDEZVOUS").expect(
                "EFA_PROBE_RENDEZVOUS must name a shared-filesystem dir both nodes can see",
            ));
        std::fs::create_dir_all(&rdvz).expect("create rendezvous directory");

        let host = std::env::var("SLURMD_NODENAME")
            .or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_else(|_| "?".to_string());
        let role_str = match role {
            Role::Responder => "responder",
            Role::Initiator => "initiator",
        };

        let Some(nic) = first_efa_nic() else {
            println!(
                "SKIP efa_srd_send_completion_locality_xnode [{role_str}@{host}]: no EFA device \
                 visible (no /dev/infiniband verbs stack on this node)."
            );
            return;
        };
        println!(
            "[{role_str}@{host}] cross-node SRD probe on `{nic}`, rendezvous={}",
            rdvz.display()
        );

        let config = probe_config();
        let mut device = IbvDevice::<EfaDevice>::try_open(&nic, config.clone())
            .expect("EFA device from IbvDevice::list should open");
        let domain = device
            .get_or_create_domain("srd-xnode-probe")
            .expect("EFA domain creation should succeed");

        let timeout = Duration::from_secs(120);

        let responder_ep = rdvz.join("responder.json");
        let initiator_ep = rdvz.join("initiator.json");
        let responder_ready = rdvz.join("responder_ready.flag");
        let done = rdvz.join("done.flag");

        match role {
            Role::Responder => {
                // Tolerate a reused directory: clear the files this side owns
                // (and the initiator's done-signal) before republishing.
                for stale in [&responder_ep, &responder_ready, &done] {
                    let _ = std::fs::remove_file(stale);
                }

                let (dst_mem, dst_view) = register_zeroed(domain);
                let mut qp = make_srd_qp(domain, &config);
                let my_info = qp.get_qp_info().expect("responder QP info");
                publish(
                    &responder_ep,
                    &Endpoint {
                        qp: my_info,
                        buf: IbvRemoteMemoryRegionView::from(&dst_view),
                    },
                );
                println!("[{role_str}@{host}] published endpoint; awaiting initiator");
                let init_info: IbvQpInfo = await_json(&initiator_ep, timeout);
                qp.connect(&init_info)
                    .expect("connect responder -> initiator");
                // Signal that our QP now holds an address handle back to the
                // initiator, so its first WRITE isn't rejected as UNKNOWN_PEER.
                publish(&responder_ready, &"ready");
                println!(
                    "[{role_str}@{host}] connected; holding QP/MR open until the initiator finishes"
                );
                await_flag(&done, timeout);

                // Cross-check from the far side: the control WRITE's bytes
                // should be present in our memory; the bad-rkey and OOB WRITEs
                // targeted an invalid key/address and must not have landed.
                let mut landed = vec![0u8; PROBE_LEN];
                // SAFETY: sole handle to this host allocation; the initiator has
                // signalled it is done, so no WRITE is still in flight.
                unsafe { dst_mem.read_at(0, &mut landed) }.expect("read destination");
                let want: Vec<u8> = (0..PROBE_LEN).map(probe_pattern).collect();
                if landed == want {
                    println!("[{role_str}@{host}] control-WRITE bytes present in local memory: OK");
                } else {
                    println!(
                        "[{role_str}@{host}] NOTE: destination did not match the control pattern at \
                         read time (unexpected)."
                    );
                }
                println!("[{role_str}@{host}] done.");
            }
            Role::Initiator => {
                let _ = std::fs::remove_file(&initiator_ep);

                let (_src_mem, src_view) = register_pattern_src(domain);
                // A local buffer to READ the remote back into, to prove the
                // control WRITE actually landed on the *other* host.
                let (readback_mem, readback_view) = register_zeroed(domain);
                let mut qp = make_srd_qp(domain, &config);
                let my_info = qp.get_qp_info().expect("initiator QP info");

                println!("[{role_str}@{host}] awaiting responder endpoint");
                let remote: Endpoint = await_json(&responder_ep, timeout);
                publish(&initiator_ep, &my_info);
                qp.connect(&remote.qp)
                    .expect("connect initiator -> responder");
                await_flag(&responder_ready, timeout);
                println!(
                    "[{role_str}@{host}] connected to responder; issuing WRITEs across the fabric"
                );

                let src = src_view.clone();
                let good_dst = remote.buf; // the responder's buffer, on the other host

                // 1. Control WRITE, then an RDMA READ-back of the remote buffer:
                //    a success completion whose bytes we can read back from the
                //    other host anchors what "success" means across nodes.
                let control = write_and_drain(&mut qp, good_dst.clone(), src.clone(), timeout);
                assert!(
                    control.is_success(),
                    "control WRITE should succeed, got {control:?}"
                );
                qp.get(readback_view.clone(), good_dst.clone())
                    .expect("post readback READ");
                let readback = drain_one(&mut qp, timeout);
                assert!(
                    readback.is_success(),
                    "readback READ should succeed, got {readback:?}"
                );
                let mut got = vec![0u8; PROBE_LEN];
                // SAFETY: sole handle to this host allocation; the READ above
                // has completed, so nothing is still writing into it.
                unsafe { readback_mem.read_at(0, &mut got) }.expect("read readback buffer");
                let want: Vec<u8> = (0..PROBE_LEN).map(probe_pattern).collect();
                assert_eq!(got, want, "control WRITE did not land on the remote host");
                println!(
                    "[{role_str}@{host}] [1] control WRITE + readback READ across nodes: {PROBE_LEN} bytes \
                     verified on the remote host"
                );

                // 2 & 3. Inject the responder-only faults across the fabric.
                let bad_rkey = probe_remote_faults(
                    &mut qp,
                    &good_dst,
                    &src,
                    timeout,
                    &format!("[{role_str}@{host}] "),
                );

                // Release the responder now that every WRITE has completed,
                // before the (possibly panicking) verdict assertion -- so a
                // failed hypothesis check doesn't strand it waiting on `done`.
                publish(&done, &"done");

                assert_not_local_only(&bad_rkey);
            }
        }
    }
}
