# plate

Rust library for writing simpler Vulkan code

[![crates.io][crates-badge]][crates-url]
[![build][build-badge]][build-url]
[![docs][docs-badge]][docs-url]
[![license][license-badge]][license-url]

[crates-badge]: https://img.shields.io/crates/v/plate
[crates-url]: https://crates.io/crates/plate
[build-badge]: https://img.shields.io/gitlab/pipeline-status/pcasotti/plate
[build-url]: https://gitlab.com/pcasotti/plate/-/pipelines
[docs-badge]: https://img.shields.io/docsrs/plate
[docs-url]: https://docs.rs/plate/0.1.5/plate/
[license-badge]: https://img.shields.io/crates/l/plate
[license-url]: https://github.com/pcasotti/plate/blob/main/LICENSE

## Installation

Add the library to your Cargo.toml file:
```toml
[dependencies]
plate = "0.3.0"
```

## Example

Example code is available in the examples directory.

Use cargo to run the examples:
```shell
cargo run --example triangle
```

## Features

- Easy initialization.
- Easy to use index and vertex buffers.
- Simple buffer creation and manipulation of data.
- Automatic buffer padding to device limits.
- Simple image creation.
- Ergonomic descriptor creation.
- Dynamic descriptor support.
