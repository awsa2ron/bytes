use core::iter::FromIterator;
use core::ops::{Deref, RangeBounds};
use core::{cmp, fmt, hash, mem, ptr, slice, usize};

use alloc::{borrow::Borrow, boxed::Box, string::String, vec::Vec};

use crate::buf::IntoIter;
#[allow(unused)]
use crate::loom::sync::atomic::AtomicMut;
use crate::loom::sync::atomic::{self, AtomicPtr, AtomicUsize, Ordering};
use crate::Buf;

/// A cheaply cloneable and sliceable chunk of contiguous memory.
///
/// `rBytes` is an efficient container for storing and operating on contiguous
/// slices of memory. It is intended for use primarily in networking code, but
/// could have applications elsewhere as well.
///
/// `rBytes` values facilitate zero-copy network programming by allowing multiple
/// `rBytes` objects to point to the same underlying memory.
///
/// `rBytes` does not have a single implementation. It is an interface, whose
/// exact behavior is implemented through dynamic dispatch in several underlying
/// implementations of `rBytes`.
///
/// All `rBytes` implementations must fulfill the following requirements:
/// - They are cheaply cloneable and thereby shareable between an unlimited amount
///   of components, for example by modifying a reference count.
/// - Instances can be sliced to refer to a subset of the the original buffer.
///
/// ```
/// use bytes::rBytes;
///
/// let mut mem = rBytes::from("Hello world");
/// let a = mem.slice(0..5);
///
/// assert_eq!(a, "Hello");
///
/// let b = mem.split_to(6);
///
/// assert_eq!(mem, "world");
/// assert_eq!(b, "Hello ");
/// ```
///
/// # Memory layout
///
/// The `rBytes` struct itself is fairly small, limited to 4 `usize` fields used
/// to track information about which segment of the underlying memory the
/// `rBytes` handle has access to.
///
/// `rBytes` keeps both a pointer to the shared state containing the full memory
/// slice and a pointer to the start of the region visible by the handle.
/// `rBytes` also tracks the length of its view into the memory.
///
/// # Sharing
///
/// `rBytes` contains a vtable, which allows implementations of `rBytes` to define
/// how sharing/cloneing is implemented in detail.
/// When `rBytes::clone()` is called, `rBytes` will call the vtable function for
/// cloning the backing storage in order to share it behind between multiple
/// `rBytes` instances.
///
/// For `rBytes` implementations which refer to constant memory (e.g. created
/// via `rBytes::from_static()`) the cloning implementation will be a no-op.
///
/// For `rBytes` implementations which point to a reference counted shared storage
/// (e.g. an `Arc<[u8]>`), sharing will be implemented by increasing the
/// the reference count.
///
/// Due to this mechanism, multiple `rBytes` instances may point to the same
/// shared memory region.
/// Each `rBytes` instance can point to different sections within that
/// memory region, and `rBytes` instances may or may not have overlapping views
/// into the memory.
///
/// The following diagram visualizes a scenario where 2 `rBytes` instances make
/// use of an `Arc`-based backing storage, and provide access to different views:
///
/// ```text
///
///    Arc ptrs                   +---------+
///    ________________________ / | rBytes 2 |
///   /                           +---------+
///  /          +-----------+     |         |
/// |_________/ |  rBytes 1  |     |         |
/// |           +-----------+     |         |
/// |           |           | ___/ data     | tail
/// |      data |      tail |/              |
/// v           v           v               v
/// +-----+---------------------------------+-----+
/// | Arc |     |           |               |     |
/// +-----+---------------------------------+-----+
/// ```
pub struct rBytes {
    ptr: *const u8,
    len: usize,
    // inlined "trait object"
    data: AtomicPtr<()>,
    vtable: &'static rVtable,
}

pub(crate) struct rVtable {
    /// fn(data, ptr, len)
    pub clone: unsafe fn(&AtomicPtr<()>, *const u8, usize) -> rBytes,
    /// fn(data, ptr, len)
    pub drop: unsafe fn(&mut AtomicPtr<()>, *const u8, usize),
}

impl rBytes {
    /// Creates a new empty `rBytes`.
    ///
    /// This will not allocate and the returned `rBytes` handle will be empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::rBytes;
    ///
    /// let b = rBytes::new();
    /// assert_eq!(&b[..], b"");
    /// ```
    #[inline]
    #[cfg(not(all(loom, test)))]
    pub const fn new() -> rBytes {
        // Make it a named const to work around
        // "unsizing casts are not allowed in const fn"
        const EMPTY: &[u8] = &[];
        rBytes::from_static(EMPTY)
    }

    #[cfg(all(loom, test))]
    pub fn new() -> rBytes {
        const EMPTY: &[u8] = &[];
        rBytes::from_static(EMPTY)
    }

    /// Creates a new `rBytes` from a static slice.
    ///
    /// The returned `rBytes` will point directly to the static slice. There is
    /// no allocating or copying.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::rBytes;
    ///
    /// let b = rBytes::from_static(b"hello");
    /// assert_eq!(&b[..], b"hello");
    /// ```
    #[inline]
    #[cfg(not(all(loom, test)))]
    pub const fn from_static(bytes: &'static [u8]) -> rBytes {
        rBytes {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
            data: AtomicPtr::new(ptr::null_mut()),
            vtable: &STATIC_VTABLE,
        }
    }

    #[cfg(all(loom, test))]
    pub fn from_static(bytes: &'static [u8]) -> rBytes {
        rBytes {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
            data: AtomicPtr::new(ptr::null_mut()),
            vtable: &STATIC_VTABLE,
        }
    }

    /// Returns the number of bytes contained in this `rBytes`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::rBytes;
    ///
    /// let b = rBytes::from(&b"hello"[..]);
    /// assert_eq!(b.len(), 5);
    /// ```
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the `rBytes` has a length of 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::rBytes;
    ///
    /// let b = rBytes::new();
    /// assert!(b.is_empty());
    /// ```
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Creates `rBytes` instance from slice, by copying it.
    pub fn copy_from_slice(data: &[u8]) -> Self {
        data.to_vec().into()
    }

    /// Returns a slice of self for the provided range.
    ///
    /// This will increment the reference count for the underlying memory and
    /// return a new `rBytes` handle set to the slice.
    ///
    /// This operation is `O(1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::rBytes;
    ///
    /// let a = rBytes::from(&b"hello world"[..]);
    /// let b = a.slice(2..5);
    ///
    /// assert_eq!(&b[..], b"llo");
    /// ```
    ///
    /// # Panics
    ///
    /// Requires that `begin <= end` and `end <= self.len()`, otherwise slicing
    /// will panic.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> rBytes {
        use core::ops::Bound;

        let len = self.len();

        let begin = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n + 1,
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(&n) => n.checked_add(1).expect("out of range"),
            Bound::Excluded(&n) => n,
            Bound::Unbounded => len,
        };

        assert!(
            begin <= end,
            "range start must not be greater than end: {:?} <= {:?}",
            begin,
            end,
        );
        assert!(
            end <= len,
            "range end out of bounds: {:?} <= {:?}",
            end,
            len,
        );

        if end == begin {
            return rBytes::new();
        }

        let mut ret = self.clone();

        ret.len = end - begin;
        ret.ptr = unsafe { ret.ptr.offset(begin as isize) };

        ret
    }

    /// Returns a slice of self that is equivalent to the given `subset`.
    ///
    /// When processing a `rBytes` buffer with other tools, one often gets a
    /// `&[u8]` which is in fact a slice of the `rBytes`, i.e. a subset of it.
    /// This function turns that `&[u8]` into another `rBytes`, as if one had
    /// called `self.slice()` with the offsets that correspond to `subset`.
    ///
    /// This operation is `O(1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::rBytes;
    ///
    /// let bytes = rBytes::from(&b"012345678"[..]);
    /// let as_slice = bytes.as_ref();
    /// let subset = &as_slice[2..6];
    /// let subslice = bytes.slice_ref(&subset);
    /// assert_eq!(&subslice[..], b"2345");
    /// ```
    ///
    /// # Panics
    ///
    /// Requires that the given `sub` slice is in fact contained within the
    /// `rBytes` buffer; otherwise this function will panic.
    pub fn slice_ref(&self, subset: &[u8]) -> rBytes {
        // Empty slice and empty rBytes may have their pointers reset
        // so explicitly allow empty slice to be a subslice of any slice.
        if subset.is_empty() {
            return rBytes::new();
        }

        let bytes_p = self.as_ptr() as usize;
        let bytes_len = self.len();

        let sub_p = subset.as_ptr() as usize;
        let sub_len = subset.len();

        assert!(
            sub_p >= bytes_p,
            "subset pointer ({:p}) is smaller than self pointer ({:p})",
            sub_p as *const u8,
            bytes_p as *const u8,
        );
        assert!(
            sub_p + sub_len <= bytes_p + bytes_len,
            "subset is out of bounds: self = ({:p}, {}), subset = ({:p}, {})",
            bytes_p as *const u8,
            bytes_len,
            sub_p as *const u8,
            sub_len,
        );

        let sub_offset = sub_p - bytes_p;

        self.slice(sub_offset..(sub_offset + sub_len))
    }

    /// Splits the bytes into two at the given index.
    ///
    /// Afterwards `self` contains elements `[0, at)`, and the returned `rBytes`
    /// contains elements `[at, len)`.
    ///
    /// This is an `O(1)` operation that just increases the reference count and
    /// sets a few indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::rBytes;
    ///
    /// let mut a = rBytes::from(&b"hello world"[..]);
    /// let b = a.split_off(5);
    ///
    /// assert_eq!(&a[..], b"hello");
    /// assert_eq!(&b[..], b" world");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
    #[must_use = "consider rBytes::truncate if you don't need the other half"]
    pub fn split_off(&mut self, at: usize) -> rBytes {
        assert!(
            at <= self.len(),
            "split_off out of bounds: {:?} <= {:?}",
            at,
            self.len(),
        );

        if at == self.len() {
            return rBytes::new();
        }

        if at == 0 {
            return mem::replace(self, rBytes::new());
        }

        let mut ret = self.clone();

        self.len = at;

        unsafe { ret.inc_start(at) };

        ret
    }

    /// Splits the bytes into two at the given index.
    ///
    /// Afterwards `self` contains elements `[at, len)`, and the returned
    /// `rBytes` contains elements `[0, at)`.
    ///
    /// This is an `O(1)` operation that just increases the reference count and
    /// sets a few indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::rBytes;
    ///
    /// let mut a = rBytes::from(&b"hello world"[..]);
    /// let b = a.split_to(5);
    ///
    /// assert_eq!(&a[..], b" world");
    /// assert_eq!(&b[..], b"hello");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
    #[must_use = "consider rBytes::advance if you don't need the other half"]
    pub fn split_to(&mut self, at: usize) -> rBytes {
        assert!(
            at <= self.len(),
            "split_to out of bounds: {:?} <= {:?}",
            at,
            self.len(),
        );

        if at == self.len() {
            return mem::replace(self, rBytes::new());
        }

        if at == 0 {
            return rBytes::new();
        }

        let mut ret = self.clone();

        unsafe { self.inc_start(at) };

        ret.len = at;
        ret
    }

    /// Shortens the buffer, keeping the first `len` bytes and dropping the
    /// rest.
    ///
    /// If `len` is greater than the buffer's current length, this has no
    /// effect.
    ///
    /// The [`split_off`] method can emulate `truncate`, but this causes the
    /// excess bytes to be returned instead of dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::rBytes;
    ///
    /// let mut buf = rBytes::from(&b"hello world"[..]);
    /// buf.truncate(5);
    /// assert_eq!(buf, b"hello"[..]);
    /// ```
    ///
    /// [`split_off`]: #method.split_off
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            // The Vec "promotable" vtables do not store the capacity,
            // so we cannot truncate while using this repr. We *have* to
            // promote using `split_off` so the capacity can be stored.
            if self.vtable as *const rVtable == &PROMOTABLE_EVEN_VTABLE
                || self.vtable as *const rVtable == &PROMOTABLE_ODD_VTABLE
            {
                drop(self.split_off(len));
            } else {
                self.len = len;
            }
        }
    }

    /// Clears the buffer, removing all data.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::rBytes;
    ///
    /// let mut buf = rBytes::from(&b"hello world"[..]);
    /// buf.clear();
    /// assert!(buf.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    #[inline]
    pub(crate) unsafe fn with_vtable(
        ptr: *const u8,
        len: usize,
        data: AtomicPtr<()>,
        vtable: &'static rVtable,
    ) -> rBytes {
        rBytes {
            ptr,
            len,
            data,
            vtable,
        }
    }

    // private

    #[inline]
    fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline]
    unsafe fn inc_start(&mut self, by: usize) {
        // should already be asserted, but debug assert for tests
        debug_assert!(self.len >= by, "internal: inc_start out of bounds");
        self.len -= by;
        self.ptr = self.ptr.offset(by as isize);
    }
    #[inline]
    unsafe fn dec_end(&mut self, by: usize) {
        // should already be asserted, but debug assert for tests
        debug_assert!(self.len >= by, "internal: inc_start out of bounds");
        self.len -= by;
        // self.ptr = self.ptr.offset(by as isize);
    }
}

// rVtable must enforce this behavior
unsafe impl Send for rBytes {}
unsafe impl Sync for rBytes {}

impl Drop for rBytes {
    #[inline]
    fn drop(&mut self) {
        unsafe { (self.vtable.drop)(&mut self.data, self.ptr, self.len) }
    }
}

impl Clone for rBytes {
    #[inline]
    fn clone(&self) -> rBytes {
        unsafe { (self.vtable.clone)(&self.data, self.ptr, self.len) }
    }
}

impl Buf for rBytes {
    #[inline]
    fn remaining(&self) -> usize {
        self.len()
    }

    #[inline]
    fn chunk(&self) -> &[u8] {
        self.as_slice()
    }

    #[inline]
    fn advance(&mut self, cnt: usize) {
        assert!(
            cnt <= self.len(),
            "cannot advance past `remaining`: {:?} <= {:?}",
            cnt,
            self.len(),
        );

        unsafe {
            self.dec_end(cnt);
        }
    }
    fn get_u8(&mut self) -> u8 {
        assert!(self.remaining() >= 1);
        let ret = self.chunk()[self.len-1];
        self.advance(1);
        ret
    }

    fn copy_to_bytes(&mut self, len: usize) -> crate::Bytes {
        // if len == self.remaining() {
        //     core::mem::replace(self, Bytes::new())
        // } else {
        //     let ret = self.slice(..len);
        //     self.advance(len);
        //     ret
        // }
            crate::Bytes::new()
    }
}

impl Deref for rBytes {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for rBytes {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl hash::Hash for rBytes {
    fn hash<H>(&self, state: &mut H)
    where
        H: hash::Hasher,
    {
        self.as_slice().hash(state);
    }
}

impl Borrow<[u8]> for rBytes {
    fn borrow(&self) -> &[u8] {
        self.as_slice()
    }
}

impl IntoIterator for rBytes {
    type Item = u8;
    type IntoIter = IntoIter<rBytes>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<'a> IntoIterator for &'a rBytes {
    type Item = &'a u8;
    type IntoIter = core::slice::Iter<'a, u8>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().into_iter()
    }
}

impl FromIterator<u8> for rBytes {
    fn from_iter<T: IntoIterator<Item = u8>>(into_iter: T) -> Self {
        Vec::from_iter(into_iter).into()
    }
}

// impl Eq

impl PartialEq for rBytes {
    fn eq(&self, other: &rBytes) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialOrd for rBytes {
    fn partial_cmp(&self, other: &rBytes) -> Option<cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl Ord for rBytes {
    fn cmp(&self, other: &rBytes) -> cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl Eq for rBytes {}

impl PartialEq<[u8]> for rBytes {
    fn eq(&self, other: &[u8]) -> bool {
        self.as_slice() == other
    }
}

impl PartialOrd<[u8]> for rBytes {
    fn partial_cmp(&self, other: &[u8]) -> Option<cmp::Ordering> {
        self.as_slice().partial_cmp(other)
    }
}

impl PartialEq<rBytes> for [u8] {
    fn eq(&self, other: &rBytes) -> bool {
        *other == *self
    }
}

impl PartialOrd<rBytes> for [u8] {
    fn partial_cmp(&self, other: &rBytes) -> Option<cmp::Ordering> {
        <[u8] as PartialOrd<[u8]>>::partial_cmp(self, other)
    }
}

impl PartialEq<str> for rBytes {
    fn eq(&self, other: &str) -> bool {
        self.as_slice() == other.as_bytes()
    }
}

impl PartialOrd<str> for rBytes {
    fn partial_cmp(&self, other: &str) -> Option<cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_bytes())
    }
}

impl PartialEq<rBytes> for str {
    fn eq(&self, other: &rBytes) -> bool {
        *other == *self
    }
}

impl PartialOrd<rBytes> for str {
    fn partial_cmp(&self, other: &rBytes) -> Option<cmp::Ordering> {
        <[u8] as PartialOrd<[u8]>>::partial_cmp(self.as_bytes(), other)
    }
}

impl PartialEq<Vec<u8>> for rBytes {
    fn eq(&self, other: &Vec<u8>) -> bool {
        *self == &other[..]
    }
}

impl PartialOrd<Vec<u8>> for rBytes {
    fn partial_cmp(&self, other: &Vec<u8>) -> Option<cmp::Ordering> {
        self.as_slice().partial_cmp(&other[..])
    }
}

impl PartialEq<rBytes> for Vec<u8> {
    fn eq(&self, other: &rBytes) -> bool {
        *other == *self
    }
}

impl PartialOrd<rBytes> for Vec<u8> {
    fn partial_cmp(&self, other: &rBytes) -> Option<cmp::Ordering> {
        <[u8] as PartialOrd<[u8]>>::partial_cmp(self, other)
    }
}

impl PartialEq<String> for rBytes {
    fn eq(&self, other: &String) -> bool {
        *self == &other[..]
    }
}

impl PartialOrd<String> for rBytes {
    fn partial_cmp(&self, other: &String) -> Option<cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_bytes())
    }
}

impl PartialEq<rBytes> for String {
    fn eq(&self, other: &rBytes) -> bool {
        *other == *self
    }
}

impl PartialOrd<rBytes> for String {
    fn partial_cmp(&self, other: &rBytes) -> Option<cmp::Ordering> {
        <[u8] as PartialOrd<[u8]>>::partial_cmp(self.as_bytes(), other)
    }
}

impl PartialEq<rBytes> for &[u8] {
    fn eq(&self, other: &rBytes) -> bool {
        *other == *self
    }
}

impl PartialOrd<rBytes> for &[u8] {
    fn partial_cmp(&self, other: &rBytes) -> Option<cmp::Ordering> {
        <[u8] as PartialOrd<[u8]>>::partial_cmp(self, other)
    }
}

impl PartialEq<rBytes> for &str {
    fn eq(&self, other: &rBytes) -> bool {
        *other == *self
    }
}

impl PartialOrd<rBytes> for &str {
    fn partial_cmp(&self, other: &rBytes) -> Option<cmp::Ordering> {
        <[u8] as PartialOrd<[u8]>>::partial_cmp(self.as_bytes(), other)
    }
}

impl<'a, T: ?Sized> PartialEq<&'a T> for rBytes
where
    rBytes: PartialEq<T>,
{
    fn eq(&self, other: &&'a T) -> bool {
        *self == **other
    }
}

impl<'a, T: ?Sized> PartialOrd<&'a T> for rBytes
where
    rBytes: PartialOrd<T>,
{
    fn partial_cmp(&self, other: &&'a T) -> Option<cmp::Ordering> {
        self.partial_cmp(&**other)
    }
}

// impl From

impl Default for rBytes {
    #[inline]
    fn default() -> rBytes {
        rBytes::new()
    }
}

impl From<&'static [u8]> for rBytes {
    fn from(slice: &'static [u8]) -> rBytes {
        rBytes::from_static(slice)
    }
}

impl From<&'static str> for rBytes {
    fn from(slice: &'static str) -> rBytes {
        rBytes::from_static(slice.as_bytes())
    }
}

impl From<Vec<u8>> for rBytes {
    fn from(vec: Vec<u8>) -> rBytes {
        let slice = vec.into_boxed_slice();
        slice.into()
    }
}

impl From<Box<[u8]>> for rBytes {
    fn from(slice: Box<[u8]>) -> rBytes {
        // Box<[u8]> doesn't contain a heap allocation for empty slices,
        // so the pointer isn't aligned enough for the KIND_VEC stashing to
        // work.
        if slice.is_empty() {
            return rBytes::new();
        }

        let len = slice.len();
        let ptr = Box::into_raw(slice) as *mut u8;

        if ptr as usize & 0x1 == 0 {
            let data = ptr as usize | KIND_VEC;
            rBytes {
                ptr,
                len,
                data: AtomicPtr::new(data as *mut _),
                vtable: &PROMOTABLE_EVEN_VTABLE,
            }
        } else {
            rBytes {
                ptr,
                len,
                data: AtomicPtr::new(ptr as *mut _),
                vtable: &PROMOTABLE_ODD_VTABLE,
            }
        }
    }
}

impl From<String> for rBytes {
    fn from(s: String) -> rBytes {
        rBytes::from(s.into_bytes())
    }
}

// ===== impl rVtable =====

impl fmt::Debug for rVtable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("rVtable")
            .field("clone", &(self.clone as *const ()))
            .field("drop", &(self.drop as *const ()))
            .finish()
    }
}

// ===== impl StaticVtable =====

const STATIC_VTABLE: rVtable = rVtable {
    clone: static_clone,
    drop: static_drop,
};

unsafe fn static_clone(_: &AtomicPtr<()>, ptr: *const u8, len: usize) -> rBytes {
    let slice = slice::from_raw_parts(ptr, len);
    rBytes::from_static(slice)
}

unsafe fn static_drop(_: &mut AtomicPtr<()>, _: *const u8, _: usize) {
    // nothing to drop for &'static [u8]
}

// ===== impl PromotableVtable =====

static PROMOTABLE_EVEN_VTABLE: rVtable = rVtable {
    clone: promotable_even_clone,
    drop: promotable_even_drop,
};

static PROMOTABLE_ODD_VTABLE: rVtable = rVtable {
    clone: promotable_odd_clone,
    drop: promotable_odd_drop,
};

unsafe fn promotable_even_clone(data: &AtomicPtr<()>, ptr: *const u8, len: usize) -> rBytes {
    let shared = data.load(Ordering::Acquire);
    let kind = shared as usize & KIND_MASK;

    if kind == KIND_ARC {
        shallow_clone_arc(shared as _, ptr, len)
    } else {
        debug_assert_eq!(kind, KIND_VEC);
        let buf = (shared as usize & !KIND_MASK) as *mut u8;
        shallow_clone_vec(data, shared, buf, ptr, len)
    }
}

unsafe fn promotable_even_drop(data: &mut AtomicPtr<()>, ptr: *const u8, len: usize) {
    data.with_mut(|shared| {
        let shared = *shared;
        let kind = shared as usize & KIND_MASK;

        if kind == KIND_ARC {
            release_shared(shared as *mut Shared);
        } else {
            debug_assert_eq!(kind, KIND_VEC);
            let buf = (shared as usize & !KIND_MASK) as *mut u8;
            drop(rebuild_boxed_slice(buf, ptr, len));
        }
    });
}

unsafe fn promotable_odd_clone(data: &AtomicPtr<()>, ptr: *const u8, len: usize) -> rBytes {
    let shared = data.load(Ordering::Acquire);
    let kind = shared as usize & KIND_MASK;

    if kind == KIND_ARC {
        shallow_clone_arc(shared as _, ptr, len)
    } else {
        debug_assert_eq!(kind, KIND_VEC);
        shallow_clone_vec(data, shared, shared as *mut u8, ptr, len)
    }
}

unsafe fn promotable_odd_drop(data: &mut AtomicPtr<()>, ptr: *const u8, len: usize) {
    data.with_mut(|shared| {
        let shared = *shared;
        let kind = shared as usize & KIND_MASK;

        if kind == KIND_ARC {
            release_shared(shared as *mut Shared);
        } else {
            debug_assert_eq!(kind, KIND_VEC);

            drop(rebuild_boxed_slice(shared as *mut u8, ptr, len));
        }
    });
}

unsafe fn rebuild_boxed_slice(buf: *mut u8, offset: *const u8, len: usize) -> Box<[u8]> {
    let cap = (offset as usize - buf as usize) + len;
    Box::from_raw(slice::from_raw_parts_mut(buf, cap))
}

// ===== impl SharedVtable =====

struct Shared {
    // holds vec for drop, but otherwise doesnt access it
    _vec: Vec<u8>,
    ref_cnt: AtomicUsize,
}

// Assert that the alignment of `Shared` is divisible by 2.
// This is a necessary invariant since we depend on allocating `Shared` a
// shared object to implicitly carry the `KIND_ARC` flag in its pointer.
// This flag is set when the LSB is 0.
const _: [(); 0 - mem::align_of::<Shared>() % 2] = []; // Assert that the alignment of `Shared` is divisible by 2.

static SHARED_VTABLE: rVtable = rVtable {
    clone: shared_clone,
    drop: shared_drop,
};

const KIND_ARC: usize = 0b0;
const KIND_VEC: usize = 0b1;
const KIND_MASK: usize = 0b1;

unsafe fn shared_clone(data: &AtomicPtr<()>, ptr: *const u8, len: usize) -> rBytes {
    let shared = data.load(Ordering::Relaxed);
    shallow_clone_arc(shared as _, ptr, len)
}

unsafe fn shared_drop(data: &mut AtomicPtr<()>, _ptr: *const u8, _len: usize) {
    data.with_mut(|shared| {
        release_shared(*shared as *mut Shared);
    });
}

unsafe fn shallow_clone_arc(shared: *mut Shared, ptr: *const u8, len: usize) -> rBytes {
    let old_size = (*shared).ref_cnt.fetch_add(1, Ordering::Relaxed);

    if old_size > usize::MAX >> 1 {
        crate::abort();
    }

    rBytes {
        ptr,
        len,
        data: AtomicPtr::new(shared as _),
        vtable: &SHARED_VTABLE,
    }
}

#[cold]
unsafe fn shallow_clone_vec(
    atom: &AtomicPtr<()>,
    ptr: *const (),
    buf: *mut u8,
    offset: *const u8,
    len: usize,
) -> rBytes {
    // If  the buffer is still tracked in a `Vec<u8>`. It is time to
    // promote the vec to an `Arc`. This could potentially be called
    // concurrently, so some care must be taken.

    // First, allocate a new `Shared` instance containing the
    // `Vec` fields. It's important to note that `ptr`, `len`,
    // and `cap` cannot be mutated without having `&mut self`.
    // This means that these fields will not be concurrently
    // updated and since the buffer hasn't been promoted to an
    // `Arc`, those three fields still are the components of the
    // vector.
    let vec = rebuild_boxed_slice(buf, offset, len).into_vec();
    let shared = Box::new(Shared {
        _vec: vec,
        // Initialize refcount to 2. One for this reference, and one
        // for the new clone that will be returned from
        // `shallow_clone`.
        ref_cnt: AtomicUsize::new(2),
    });

    let shared = Box::into_raw(shared);

    // The pointer should be aligned, so this assert should
    // always succeed.
    debug_assert!(
        0 == (shared as usize & KIND_MASK),
        "internal: Box<Shared> should have an aligned pointer",
    );

    // Try compare & swapping the pointer into the `arc` field.
    // `Release` is used synchronize with other threads that
    // will load the `arc` field.
    //
    // If the `compare_exchange` fails, then the thread lost the
    // race to promote the buffer to shared. The `Acquire`
    // ordering will synchronize with the `compare_exchange`
    // that happened in the other thread and the `Shared`
    // pointed to by `actual` will be visible.
    match atom.compare_exchange(ptr as _, shared as _, Ordering::AcqRel, Ordering::Acquire) {
        Ok(actual) => {
            debug_assert!(actual as usize == ptr as usize);
            // The upgrade was successful, the new handle can be
            // returned.
            rBytes {
                ptr: offset,
                len,
                data: AtomicPtr::new(shared as _),
                vtable: &SHARED_VTABLE,
            }
        }
        Err(actual) => {
            // The upgrade failed, a concurrent clone happened. Release
            // the allocation that was made in this thread, it will not
            // be needed.
            let shared = Box::from_raw(shared);
            mem::forget(*shared);

            // Buffer already promoted to shared storage, so increment ref
            // count.
            shallow_clone_arc(actual as _, offset, len)
        }
    }
}

unsafe fn release_shared(ptr: *mut Shared) {
    // `Shared` storage... follow the drop steps from Arc.
    if (*ptr).ref_cnt.fetch_sub(1, Ordering::Release) != 1 {
        return;
    }

    // This fence is needed to prevent reordering of use of the data and
    // deletion of the data.  Because it is marked `Release`, the decreasing
    // of the reference count synchronizes with this `Acquire` fence. This
    // means that use of the data happens before decreasing the reference
    // count, which happens before this fence, which happens before the
    // deletion of the data.
    //
    // As explained in the [Boost documentation][1],
    //
    // > It is important to enforce any possible access to the object in one
    // > thread (through an existing reference) to *happen before* deleting
    // > the object in a different thread. This is achieved by a "release"
    // > operation after dropping a reference (any access to the object
    // > through this reference must obviously happened before), and an
    // > "acquire" operation before deleting the object.
    //
    // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
    atomic::fence(Ordering::Acquire);

    // Drop the data
    Box::from_raw(ptr);
}

// compile-fails

/// ```compile_fail
/// use bytes::rBytes;
/// #[deny(unused_must_use)]
/// {
///     let mut b1 = rBytes::from("hello world");
///     b1.split_to(6);
/// }
/// ```
fn _split_to_must_use() {}

/// ```compile_fail
/// use bytes::rBytes;
/// #[deny(unused_must_use)]
/// {
///     let mut b1 = rBytes::from("hello world");
///     b1.split_off(6);
/// }
/// ```
fn _split_off_must_use() {}

// fuzz tests
#[cfg(all(test, loom))]
mod fuzz {
    use loom::sync::Arc;
    use loom::thread;

    use super::rBytes;
    #[test]
    fn bytes_cloning_vec() {
        loom::model(|| {
            let a = rBytes::from(b"abcdefgh".to_vec());
            let addr = a.as_ptr() as usize;

            // test the rBytes::clone is Sync by putting it in an Arc
            let a1 = Arc::new(a);
            let a2 = a1.clone();

            let t1 = thread::spawn(move || {
                let b: rBytes = (*a1).clone();
                assert_eq!(b.as_ptr() as usize, addr);
            });

            let t2 = thread::spawn(move || {
                let b: rBytes = (*a2).clone();
                assert_eq!(b.as_ptr() as usize, addr);
            });

            t1.join().unwrap();
            t2.join().unwrap();
        });
    }
}
