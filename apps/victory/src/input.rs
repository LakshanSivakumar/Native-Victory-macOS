//! Live input of the native macOS host, on the main thread beside the window's event loop: the
//! window's key and mouse events, and pads from Apple's Game Controller framework (a DualSense,
//! an Xbox pad, anything it knows), handed to `GameInput` (`victory_hle::modules::gameinput::mac`).
//!
//! - Keys go in as Windows virtual keys and set-1 scan codes. Key events are not passed on to
//!   AppKit (which would beep at each), except with Command held, so Command-Q still quits.
//! - A click in the window captures the mouse: the pointer is hidden and held, and its motion
//!   turns the right stick; left button fires (right trigger), right button aims (left trigger),
//!   middle clicks the right stick. F1, or the window losing focus, lets the mouse go.
//! - Pads are read each turn of the loop. Each slot gets the controller of that index, its player
//!   light set once.
//!
//! The Game Controller framework is reached through the Objective-C runtime, like AppKit in
//! `window.rs`.

use objc2::msg_send;
use objc2::runtime::{AnyObject, Bool, Sel};
use objc2::{class, sel};
use victory_hle::modules::gameinput::mac;

#[link(name = "GameController", kind = "framework")]
unsafe extern "C" {}

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGAssociateMouseAndMouseCursorPosition(connected: u32) -> i32;
}

/// What the loop keeps between turns.
#[derive(Default)]
pub struct Input {
    captured: bool,
    focused: bool,
    /// Controllers whose player light was set (by address).
    lit: Vec<usize>,
    /// The pads last published, to log connections once.
    connected: [bool; 4],
}

// NSEvent types.
const LEFT_DOWN: usize = 1;
const LEFT_UP: usize = 2;
const RIGHT_DOWN: usize = 3;
const RIGHT_UP: usize = 4;
const MOVED: usize = 5;
const LEFT_DRAGGED: usize = 6;
const RIGHT_DRAGGED: usize = 7;
const KEY_DOWN: usize = 10;
const KEY_UP: usize = 11;
const FLAGS_CHANGED: usize = 12;
const OTHER_DOWN: usize = 25;
const OTHER_UP: usize = 26;
const OTHER_DRAGGED: usize = 27;

/// `NSEventModifierFlagCommand`.
const COMMAND: usize = 1 << 20;

impl Input {
    /// Takes `event` if it is input for the title; false to pass it on to AppKit.
    ///
    /// # Safety
    /// On the main thread, with an `NSEvent` and our window.
    pub unsafe fn event(&mut self, window: &AnyObject, event: *mut AnyObject) -> bool {
        // SAFETY: NSEvent accessors on the main thread.
        unsafe {
            let kind: usize = msg_send![event, type];
            let flags: usize = msg_send![event, modifierFlags];
            match kind {
                KEY_DOWN | KEY_UP => {
                    if flags & COMMAND != 0 {
                        return false;
                    }
                    let code: u16 = msg_send![event, keyCode];
                    let down = kind == KEY_DOWN;
                    if down && code == 0x7A {
                        // F1: the mouse goes free.
                        self.release(window);
                    }
                    if let Some((vk, scan)) = windows_key(code) {
                        let repeat: Bool = msg_send![event, isARepeat];
                        if down && !repeat.as_bool() {
                            tracing::info!(vk, "key");
                        }
                        let code_point = if down { first_char(event) } else { 0 };
                        mac::key(mac::Key { virtual_key: vk, scan_code: scan, code_point }, down);
                    }
                    true
                }
                FLAGS_CHANGED => {
                    let code: u16 = msg_send![event, keyCode];
                    if let Some((vk, scan)) = windows_key(code) {
                        // Which flag the key holds tells up or down.
                        let bit = match code {
                            0x38 | 0x3C => 1 << 17, // shift
                            0x3B | 0x3E => 1 << 18, // control
                            0x3A | 0x3D => 1 << 19, // option
                            0x39 => 1 << 16,        // caps lock
                            _ => 0,
                        };
                        mac::key(mac::Key { virtual_key: vk, scan_code: scan, code_point: 0 }, flags & bit != 0);
                    }
                    false
                }
                MOVED | LEFT_DRAGGED | RIGHT_DRAGGED | OTHER_DRAGGED => {
                    if self.captured {
                        let dx: f64 = msg_send![event, deltaX];
                        let dy: f64 = msg_send![event, deltaY];
                        mac::mouse_moved(dx, dy);
                        return true;
                    }
                    false
                }
                LEFT_DOWN | LEFT_UP | RIGHT_DOWN | RIGHT_UP | OTHER_DOWN | OTHER_UP => {
                    let down = matches!(kind, LEFT_DOWN | RIGHT_DOWN | OTHER_DOWN);
                    if !self.captured {
                        if kind == LEFT_DOWN && self.in_content(window, event) {
                            self.capture();
                            return true;
                        }
                        return false;
                    }
                    let button = match kind {
                        LEFT_DOWN | LEFT_UP => 0,
                        RIGHT_DOWN | RIGHT_UP => 1,
                        _ => 2,
                    };
                    mac::mouse_button(button, down);
                    true
                }
                _ => false,
            }
        }
    }

    /// Each turn of the loop: focus, then the pads.
    ///
    /// # Safety
    /// On the main thread, with our window.
    pub unsafe fn tick(&mut self, window: &AnyObject) {
        // SAFETY: AppKit and Game Controller calls on the main thread.
        unsafe {
            let key: Bool = msg_send![window, isKeyWindow];
            if self.focused && !key.as_bool() {
                self.release(window);
                mac::release_all();
            }
            self.focused = key.as_bool();
            self.pads();
        }
    }

    fn capture(&mut self) {
        // SAFETY: plain CoreGraphics and AppKit calls on the main thread.
        unsafe {
            CGAssociateMouseAndMouseCursorPosition(0);
            let _: () = msg_send![class!(NSCursor), hide];
        }
        self.captured = true;
        mac::set_captured(true);
        tracing::info!("mouse captured (F1 lets it go)");
    }

    fn release(&mut self, _window: &AnyObject) {
        if !self.captured {
            return;
        }
        // SAFETY: as above.
        unsafe {
            CGAssociateMouseAndMouseCursorPosition(1);
            let _: () = msg_send![class!(NSCursor), unhide];
        }
        self.captured = false;
        mac::set_captured(false);
        mac::release_all();
        tracing::info!("mouse let go");
    }

    /// Whether a mouse event lies in the window's content (not its title bar).
    unsafe fn in_content(&self, window: &AnyObject, event: *mut AnyObject) -> bool {
        // SAFETY: AppKit accessors on the main thread.
        unsafe {
            let at: objc2_foundation::NSPoint = msg_send![event, locationInWindow];
            let view: *mut AnyObject = msg_send![window, contentView];
            let frame: objc2_foundation::NSRect = msg_send![view, frame];
            at.x >= 0.0 && at.y >= 0.0 && at.x < frame.size.width && at.y < frame.size.height
        }
    }

    /// The controllers the Game Controller framework knows, as pads 0 to 3.
    unsafe fn pads(&mut self) {
        // SAFETY: Game Controller calls on the main thread; nil checked before use.
        unsafe {
            let list: *mut AnyObject = msg_send![class!(GCController), controllers];
            let count: usize = if list.is_null() { 0 } else { msg_send![list, count] };
            for slot in 0..4 {
                let pad = if slot < count {
                    let c: *mut AnyObject = msg_send![list, objectAtIndex: slot];
                    self.light(c, slot);
                    read_pad(c)
                } else {
                    None
                };
                if pad.is_some() != self.connected[slot] {
                    self.connected[slot] = pad.is_some();
                    let name = if pad.is_some() && slot < count {
                        let c: *mut AnyObject = msg_send![list, objectAtIndex: slot];
                        let n: *mut AnyObject = msg_send![c, vendorName];
                        ns_string(n)
                    } else {
                        String::new()
                    };
                    tracing::info!(slot, connected = pad.is_some(), name, "controller");
                }
                mac::set_pad(slot, pad);
            }
        }
    }

    /// Player light `slot + 1` on controller `c`, once.
    unsafe fn light(&mut self, c: *mut AnyObject, slot: usize) {
        if c.is_null() || self.lit.contains(&(c as usize)) {
            return;
        }
        self.lit.push(c as usize);
        // SAFETY: `GCController` setters on the main thread.
        unsafe {
            let _: () = msg_send![c, setPlayerIndex: slot as isize];
        }
    }
}

/// A controller's extended gamepad as a pad; none without one.
unsafe fn read_pad(c: *mut AnyObject) -> Option<mac::Pad> {
    // SAFETY: Game Controller accessors on the main thread; each optional element checked.
    unsafe {
        if c.is_null() {
            return None;
        }
        let gp: *mut AnyObject = msg_send![c, extendedGamepad];
        if gp.is_null() {
            return None;
        }
        let pressed = |e: *mut AnyObject| -> bool {
            if e.is_null() {
                return false;
            }
            let b: Bool = msg_send![e, isPressed];
            b.as_bool()
        };
        let value = |e: *mut AnyObject| -> f32 {
            if e.is_null() {
                return 0.0;
            }
            msg_send![e, value]
        };
        let element = |sel: Sel| -> *mut AnyObject {
            let has: Bool = msg_send![gp, respondsToSelector: sel];
            if !has.as_bool() {
                return std::ptr::null_mut();
            }
            let e: *mut AnyObject = match sel {
                s if s == sel!(buttonA) => msg_send![gp, buttonA],
                s if s == sel!(buttonB) => msg_send![gp, buttonB],
                s if s == sel!(buttonX) => msg_send![gp, buttonX],
                s if s == sel!(buttonY) => msg_send![gp, buttonY],
                s if s == sel!(leftShoulder) => msg_send![gp, leftShoulder],
                s if s == sel!(rightShoulder) => msg_send![gp, rightShoulder],
                s if s == sel!(leftTrigger) => msg_send![gp, leftTrigger],
                s if s == sel!(rightTrigger) => msg_send![gp, rightTrigger],
                s if s == sel!(buttonMenu) => msg_send![gp, buttonMenu],
                s if s == sel!(buttonOptions) => msg_send![gp, buttonOptions],
                s if s == sel!(leftThumbstickButton) => msg_send![gp, leftThumbstickButton],
                s if s == sel!(rightThumbstickButton) => msg_send![gp, rightThumbstickButton],
                s if s == sel!(touchpadButton) => msg_send![gp, touchpadButton],
                s if s == sel!(dpad) => msg_send![gp, dpad],
                s if s == sel!(leftThumbstick) => msg_send![gp, leftThumbstick],
                s if s == sel!(rightThumbstick) => msg_send![gp, rightThumbstick],
                _ => std::ptr::null_mut(),
            };
            e
        };
        let axis = |stick: *mut AnyObject, y: bool| -> f32 {
            if stick.is_null() {
                return 0.0;
            }
            let a: *mut AnyObject = if y { msg_send![stick, yAxis] } else { msg_send![stick, xAxis] };
            value(a)
        };
        let dpad = element(sel!(dpad));
        let dir = |which: Sel| -> bool {
            if dpad.is_null() {
                return false;
            }
            let b: *mut AnyObject = match which {
                s if s == sel!(up) => msg_send![dpad, up],
                s if s == sel!(down) => msg_send![dpad, down],
                s if s == sel!(left) => msg_send![dpad, left],
                _ => msg_send![dpad, right],
            };
            pressed(b)
        };
        let mut buttons = 0;
        let mut set = |on: bool, bit: u32| {
            if on {
                buttons |= bit;
            }
        };
        // A DualSense's cross, circle, square and triangle are A, B, X and Y here, where an
        // Xbox pad has them.
        set(pressed(element(sel!(buttonA))), mac::A);
        set(pressed(element(sel!(buttonB))), mac::B);
        set(pressed(element(sel!(buttonX))), mac::X);
        set(pressed(element(sel!(buttonY))), mac::Y);
        set(pressed(element(sel!(leftShoulder))), mac::LEFT_SHOULDER);
        set(pressed(element(sel!(rightShoulder))), mac::RIGHT_SHOULDER);
        // Options is Menu; Create (and the touchpad's click) is View.
        set(pressed(element(sel!(buttonMenu))), mac::MENU);
        set(pressed(element(sel!(buttonOptions))), mac::VIEW);
        set(pressed(element(sel!(touchpadButton))), mac::VIEW);
        set(pressed(element(sel!(leftThumbstickButton))), mac::LEFT_STICK);
        set(pressed(element(sel!(rightThumbstickButton))), mac::RIGHT_STICK);
        set(dir(sel!(up)), mac::UP);
        set(dir(sel!(down)), mac::DOWN);
        set(dir(sel!(left)), mac::LEFT);
        set(dir(sel!(right)), mac::RIGHT);
        let (left, right) = (element(sel!(leftThumbstick)), element(sel!(rightThumbstick)));
        Some(mac::Pad {
            buttons,
            left_trigger: value(element(sel!(leftTrigger))),
            right_trigger: value(element(sel!(rightTrigger))),
            left_x: axis(left, false),
            left_y: axis(left, true),
            right_x: axis(right, false),
            right_y: axis(right, true),
        })
    }
}

/// An `NSString` as a Rust string (empty for nil).
unsafe fn ns_string(s: *mut AnyObject) -> String {
    if s.is_null() {
        return String::new();
    }
    // SAFETY: an NSString; its UTF-8 bytes live while it does.
    unsafe {
        let p: *const std::ffi::c_char = msg_send![s, UTF8String];
        if p.is_null() {
            return String::new();
        }
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

/// The first character a key event types (0 for none).
unsafe fn first_char(event: *mut AnyObject) -> u32 {
    // SAFETY: NSEvent accessors on the main thread.
    unsafe {
        let s: *mut AnyObject = msg_send![event, characters];
        ns_string(s).chars().next().map_or(0, |c| c as u32)
    }
}

/// A Mac key code (`kVK_*`) as a Windows virtual key and set-1 scan code (0xE0xx when
/// extended); none for keys the title has no use for.
pub fn windows_key(code: u16) -> Option<(u8, u32)> {
    Some(match code {
        0x00 => (b'A', 0x1E),
        0x01 => (b'S', 0x1F),
        0x02 => (b'D', 0x20),
        0x03 => (b'F', 0x21),
        0x04 => (b'H', 0x23),
        0x05 => (b'G', 0x22),
        0x06 => (b'Z', 0x2C),
        0x07 => (b'X', 0x2D),
        0x08 => (b'C', 0x2E),
        0x09 => (b'V', 0x2F),
        0x0B => (b'B', 0x30),
        0x0C => (b'Q', 0x10),
        0x0D => (b'W', 0x11),
        0x0E => (b'E', 0x12),
        0x0F => (b'R', 0x13),
        0x10 => (b'Y', 0x15),
        0x11 => (b'T', 0x14),
        0x12 => (b'1', 0x02),
        0x13 => (b'2', 0x03),
        0x14 => (b'3', 0x04),
        0x15 => (b'4', 0x05),
        0x16 => (b'6', 0x07),
        0x17 => (b'5', 0x06),
        0x18 => (0xBB, 0x0D), // =
        0x19 => (b'9', 0x0A),
        0x1A => (b'7', 0x08),
        0x1B => (0xBD, 0x0C), // -
        0x1C => (b'8', 0x09),
        0x1D => (b'0', 0x0B),
        0x1E => (0xDD, 0x1B), // ]
        0x1F => (b'O', 0x18),
        0x20 => (b'U', 0x16),
        0x21 => (0xDB, 0x1A), // [
        0x22 => (b'I', 0x17),
        0x23 => (b'P', 0x19),
        0x24 => (0x0D, 0x1C), // Return
        0x25 => (b'L', 0x26),
        0x26 => (b'J', 0x24),
        0x27 => (0xDE, 0x28), // '
        0x28 => (b'K', 0x25),
        0x29 => (0xBA, 0x27), // ;
        0x2A => (0xDC, 0x2B), // backslash
        0x2B => (0xBC, 0x33), // ,
        0x2C => (0xBF, 0x35), // /
        0x2D => (b'N', 0x31),
        0x2E => (b'M', 0x32),
        0x2F => (0xBE, 0x34), // .
        0x30 => (0x09, 0x0F), // Tab
        0x31 => (0x20, 0x39), // Space
        0x32 => (0xC0, 0x29), // `
        0x33 => (0x08, 0x0E), // Delete (Backspace)
        0x35 => (0x1B, 0x01), // Escape
        0x38 => (0xA0, 0x2A), // Shift
        0x39 => (0x14, 0x3A), // Caps Lock
        0x3A => (0xA4, 0x38), // Option (Alt)
        0x3B => (0xA2, 0x1D), // Control
        0x3C => (0xA1, 0x36), // right Shift
        0x3D => (0xA5, 0xE038), // right Option
        0x3E => (0xA3, 0xE01D), // right Control
        0x4C => (0x0D, 0xE01C), // keypad Enter
        0x60 => (0x74, 0x3F), // F5
        0x61 => (0x75, 0x40), // F6
        0x62 => (0x76, 0x41), // F7
        0x63 => (0x72, 0x3D), // F3
        0x64 => (0x77, 0x42), // F8
        0x65 => (0x78, 0x43), // F9
        0x67 => (0x7A, 0x57), // F11
        0x6D => (0x79, 0x44), // F10
        0x6F => (0x7B, 0x58), // F12
        0x73 => (0x24, 0xE047), // Home
        0x74 => (0x21, 0xE049), // Page Up
        0x75 => (0x2E, 0xE053), // Forward Delete
        0x76 => (0x73, 0x3E), // F4
        0x77 => (0x23, 0xE04F), // End
        0x78 => (0x71, 0x3C), // F2
        0x79 => (0x22, 0xE051), // Page Down
        0x7A => (0x70, 0x3B), // F1
        0x7B => (0x25, 0xE04B), // Left
        0x7C => (0x27, 0xE04D), // Right
        0x7D => (0x28, 0xE050), // Down
        0x7E => (0x26, 0xE048), // Up
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::windows_key;

    #[test]
    fn mac_keys_are_windows_keys() {
        assert_eq!(windows_key(0x0D), Some((b'W', 0x11)));
        assert_eq!(windows_key(0x24), Some((0x0D, 0x1C)));
        assert_eq!(windows_key(0x31), Some((0x20, 0x39)));
        assert_eq!(windows_key(0x7E), Some((0x26, 0xE048)));
        assert_eq!(windows_key(0x37), None); // Command
    }
}
