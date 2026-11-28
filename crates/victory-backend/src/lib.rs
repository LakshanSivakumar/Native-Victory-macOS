//! The seam between victory-gpu and a host GPU API: traits only, no platform code.
//! `victory-metal` implements them on Metal; victory-gpu's native host is generic over them.

pub mod backend;
#[cfg(test)]
mod backend_tests;
pub mod resources;
#[cfg(test)]
mod resources_tests;
