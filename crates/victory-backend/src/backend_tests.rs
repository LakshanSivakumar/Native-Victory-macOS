//! Pins for the argument-buffer layout of [`crate::backend::RootArgs`]:
//! root constants appended little-endian, root views padded to 8 bytes and
//! appended as a 64-bit GPU address.

use crate::backend::RootArgs;

#[test]
fn root_constants_append_little_endian_words() {
    let a = RootArgs::default().constants(&[0x0102_0304, 0x0506_0708]);
    assert_eq!(a.0, vec![0x04, 0x03, 0x02, 0x01, 0x08, 0x07, 0x06, 0x05]);
}

#[test]
fn root_address_pads_to_eight_bytes_then_appends_the_gpu_address() {
    let a = RootArgs::default()
        .constants(&[0xAABB_CCDD])
        .address(0x1122_3344_5566_7788);
    assert_eq!(a.0.len(), 16);
    assert_eq!(&a.0[0..4], &[0xDD, 0xCC, 0xBB, 0xAA]);
    assert_eq!(&a.0[4..8], &[0, 0, 0, 0]);
    assert_eq!(
        &a.0[8..16],
        &[0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]
    );
}

#[test]
fn root_address_on_an_empty_buffer_is_just_the_address() {
    let a = RootArgs::default().address(1);
    assert_eq!(a.0.len(), 8);
    assert_eq!(&a.0[..], &[1, 0, 0, 0, 0, 0, 0, 0]);
}
