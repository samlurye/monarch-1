/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Link this crate's own targets (notably its `cargo test` binaries) against
//! the static rdma-core archives built by `monarch_cpp_static_libs`.
//!
//! `monarch_rdma`'s RDMA data path calls into libibverbs/libmlx5/libefa via
//! `rdmaxcel-sys`. Those archives are wired up with `cargo:rustc-link-arg`,
//! which — unlike `rustc-link-lib` — does NOT propagate from a dependency to
//! its dependents; it only affects the artifacts of the package that emits it.
//! `monarch_extension` re-emits it in its own `build.rs` for the final cdylib;
//! without an equivalent here, a plain `cargo test -p monarch_rdma` links with
//! every `ibv_*` / `mlx5dv_*` / `efadv_*` symbol undefined.
//!
//! Re-emitting the same directives (via the shared `build_utils` helper, which
//! reads the paths from `monarch_cpp_static_libs`'s `DEP_*` metadata) makes the
//! archives part of this crate's own test/bin/example link.
fn main() {
    build_utils::setup_cpp_static_libs();
}
