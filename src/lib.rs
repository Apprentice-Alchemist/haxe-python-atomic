use pyo3::{prelude::*, types::PyBool, PyTraverseError, PyVisit};
use std::sync::atomic::{self, AtomicPtr, Ordering};

#[pymodule(gil_used = false)]
fn haxe_atomic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AtomicBool>()?;
    m.add_class::<AtomicInt>()?;
    m.add_class::<AtomicObject>()?;
    Ok(())
}

#[pyclass(module = "haxe_atomic", frozen)]
pub struct AtomicBool {
    inner: atomic::AtomicBool,
}

#[pymethods]
impl AtomicBool {
    #[new]
    fn new(val: Bound<PyBool>) -> PyResult<Self> {
        Ok(Self {
            inner: atomic::AtomicBool::new(val.extract()?),
        })
    }

    pub fn load(&self) -> bool {
        self.inner.load(Ordering::SeqCst)
    }

    pub fn store(&self, val: bool) -> bool {
        self.inner.store(val, Ordering::SeqCst);
        val
    }

    pub fn exchange(&self, val: bool) -> bool {
        self.inner.swap(val, Ordering::SeqCst)
    }

    pub fn compare_exchange(&self, current: bool, new: bool) -> bool {
        match self
            .inner
            .compare_exchange(current, new, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(v) => v,
            Err(v) => v,
        }
    }
}
#[pyclass(module = "haxe_atomic", frozen)]
pub struct AtomicInt {
    inner: atomic::AtomicI32,
}

#[pymethods]
impl AtomicInt {
    #[new]
    fn new(val: i32) -> PyResult<Self> {
        Ok(Self {
            inner: atomic::AtomicI32::new(val),
        })
    }

    pub fn load(&self) -> i32 {
        self.inner.load(Ordering::SeqCst)
    }

    pub fn store(&self, val: i32) -> i32 {
        self.inner.store(val, Ordering::SeqCst);
        val
    }

    pub fn exchange(&self, val: i32) -> i32 {
        self.inner.swap(val, Ordering::SeqCst)
    }

    pub fn compare_exchange(&self, current: i32, new: i32) -> i32 {
        match self
            .inner
            .compare_exchange(current, new, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(v) => v,
            Err(v) => v,
        }
    }

    pub fn fetch_add(&self, val: i32) -> i32 {
        self.inner.fetch_add(val, Ordering::SeqCst)
    }

    pub fn fetch_sub(&self, val: i32) -> i32 {
        self.inner.fetch_sub(val, Ordering::SeqCst)
    }

    pub fn fetch_and(&self, val: i32) -> i32 {
        self.inner.fetch_and(val, Ordering::SeqCst)
    }

    pub fn fetch_or(&self, val: i32) -> i32 {
        self.inner.fetch_or(val, Ordering::SeqCst)
    }

    pub fn fetch_xor(&self, val: i32) -> i32 {
        self.inner.fetch_xor(val, Ordering::SeqCst)
    }
}

#[pyclass(module = "haxe_atomic", frozen)]
#[derive(Debug)]
pub struct AtomicObject {
    // Invariant: contains an owned pointer to a valid python object
    value: AtomicPtr<pyo3::ffi::PyObject>,
}

#[pymethods]
impl AtomicObject {
    #[new]
    fn new(val: Py<PyAny>) -> Self {
        Self {
            value: AtomicPtr::new(val.into_ptr()),
        }
    }

    pub fn load(&self, token: Python) -> Py<PyAny> {
        // Safety: `self.value` contains a pointer to a python object
        unsafe { Py::from_borrowed_ptr(token, self.value.load(Ordering::SeqCst)) }
    }

    pub fn store(&self, val: Bound<PyAny>) -> Py<PyAny> {
        let ret = val.clone();
        let old = self.value.swap(val.into_ptr(), Ordering::SeqCst);
        // Safety: the GIL is held and `old` is a valid pointer
        unsafe { pyo3::ffi::Py_DecRef(old) };
        ret.unbind()
    }

    pub fn exchange(&self, val: Bound<PyAny>) -> Py<PyAny> {
        let token = val.py();
        let old_value = self.value.swap(val.into_ptr(), Ordering::SeqCst);
        // Safety: `self.value` always contains a pointer to a python object
        // so `old_value` is valid pointer
        // `old_value` is not stored in `self.value` anymore and is thus owned
        unsafe { Py::from_owned_ptr(token, old_value) }
    }

    pub fn compare_exchange<'a>(
        &'a self,
        expected: Bound<'a, PyAny>,
        mut desired: Bound<'a, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let py = expected.py();
        let mut orig: Bound<PyAny> = self.load(py).into_bound(py);
        while orig.eq(&expected)? {
            let desired_ptr = desired.into_ptr();
            match self.value.compare_exchange(
                orig.as_ptr(),
                desired_ptr,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(orig) => return Ok(unsafe { Py::from_owned_ptr(py, orig) }),
                Err(cur_val) => {
                    // Safety: `cur_val` is a pointer to a valid python object (see invariant of self.value)
                    orig = unsafe { Bound::from_borrowed_ptr(py, cur_val) };
                    // Safety: `desired` has not been stored in self.value and is thus still owned
                    desired = unsafe { Bound::from_owned_ptr(py, cur_val) };
                }
            }
        }
        Ok(orig.unbind())
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        // The following must use a method that does not increment the ref count
        // otherwise the cycle collector may fail to detect cycles
        let object = std::mem::ManuallyDrop::new(unsafe {
            Py::<PyAny>::from_owned_ptr_or_opt(
                Python::assume_gil_acquired(),
                self.value.load(Ordering::SeqCst),
            )
        });
        visit.call(&*object)?;
        Ok(())
    }

    fn __clear__(&self) {
        // Clear reference and decrement its ref counter
        let ptr = self.value.swap(core::ptr::null_mut(), Ordering::SeqCst);
        // Safety: the GIL is held and `ptr` is either valid or null
        unsafe {
            pyo3::ffi::Py_DecRef(ptr);
        }
    }
}

impl Drop for AtomicObject {
    fn drop(&mut self) {
        self.__clear__();
    }
}
