//! Thin Rust bindings to ChessTB WDL and DTZ probes.
use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    io,
    path::Path,
    ptr::NonNull,
};

#[derive(Debug, PartialEq, Eq)]
pub enum Wdl {
    Loss,
    BlessedLoss,
    Draw,
    CursedWin,
    Win,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RootMove {
    pub dtz: i32,
    pub from: u8,
    pub to: u8,
    /// 0 = none, 1 = knight, 2 = bishop, 3 = rook, 4 = queen.
    pub promotion: u8,
}

pub struct Tables(NonNull<c_void>);

impl Tables {
    pub fn open(directory: &Path) -> io::Result<Self> {
        let path = CString::new(directory.as_os_str().as_encoded_bytes())?;
        let mut error = [0; 512];
        // Both buffers live through the call; ownership of the handle passes to Tables.
        let handle = unsafe { ce5_tb_open(path.as_ptr(), error.as_mut_ptr(), error.len()) };
        NonNull::new(handle)
            .map(Self)
            .ok_or_else(|| native_error(&error))
    }

    pub fn largest(&self) -> usize {
        // The handle remains valid for the lifetime of self.
        unsafe { ce5_tb_largest(self.0.as_ptr()) }
    }

    pub fn wdl(&mut self, fen: &CStr, ep: u8, clock: u8) -> io::Result<Wdl> {
        let mut error = [0; 512];
        // The owned handle and borrowed buffers remain valid throughout the probe.
        let result = unsafe {
            ce5_tb_wdl(
                self.0.as_ptr(),
                fen.as_ptr(),
                u32::from(ep),
                u32::from(clock),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if result < 0 {
            Err(io::Error::other(format!(
                "{}: {}",
                fen.to_string_lossy(),
                native_error(&error)
            )))
        } else {
            match result {
                0 => Ok(Wdl::Loss),
                1 => Ok(Wdl::BlessedLoss),
                2 => Ok(Wdl::Draw),
                3 => Ok(Wdl::CursedWin),
                4 => Ok(Wdl::Win),
                _ => Err(io::Error::other("invalid ChessTB WDL class")),
            }
        }
    }

    /// Probe all legal root moves with ChessTB's DTZ ranker.
    pub fn root_dtz(
        &mut self,
        fen: &CStr,
        ep: u8,
        clock: u8,
        use_rule50: bool,
        has_repeated: bool,
    ) -> io::Result<Vec<RootMove>> {
        let mut moves = [RootMove::default(); 256];
        let mut error = [0; 512];
        // Handle and buffers remain valid throughout the native call.
        let count = unsafe {
            ce5_tb_root_dtz(
                self.0.as_ptr(),
                fen.as_ptr(),
                u32::from(ep),
                u32::from(clock),
                use_rule50,
                has_repeated,
                moves.as_mut_ptr(),
                moves.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if count < 0 {
            return Err(native_error(&error));
        }
        Ok(moves[..count as usize].to_vec())
    }
}

impl Drop for Tables {
    fn drop(&mut self) {
        // This handle was allocated by ce5_tb_open and is freed exactly once.
        unsafe { ce5_tb_close(self.0.as_ptr()) };
    }
}

fn native_error(buffer: &[c_char]) -> io::Error {
    // The zero-initialized buffer is populated with snprintf, which terminates it.
    let message = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_string_lossy();
    io::Error::other(format!("ChessTB: {message}"))
}

unsafe extern "C" {
    fn ce5_tb_root_dtz(
        tables: *mut c_void,
        fen: *const c_char,
        ep: u32,
        clock: u32,
        use_rule50: bool,
        has_repeated: bool,
        output: *mut RootMove,
        output_capacity: usize,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn ce5_tb_open(directory: *const c_char, error: *mut c_char, capacity: usize) -> *mut c_void;
    fn ce5_tb_close(tables: *mut c_void);
    fn ce5_tb_largest(tables: *mut c_void) -> usize;
    fn ce5_tb_wdl(
        tables: *mut c_void,
        fen: *const c_char,
        ep: u32,
        clock: u32,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
}

#[cfg(test)]
mod tests;
