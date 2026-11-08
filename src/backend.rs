use std::{
    cell::RefCell,
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

pub trait Backend: Clone + 'static {
    type Storage: Clone;

    fn scalar(x: f32) -> Self::Storage;
    fn zeros(shape: impl IntoShape) -> Self::Storage;
    fn ones(shape: impl IntoShape) -> Self::Storage;
    fn randn(shape: impl IntoShape) -> Self::Storage;
    fn shape(a: &Self::Storage) -> &[usize];
    fn add(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
    fn sub(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
    fn mul(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
    fn div(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
    fn matmul(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;

    fn exp(a: &Self::Storage) -> Self::Storage;
    fn log(a: &Self::Storage) -> Self::Storage;
    fn powf(a: &Self::Storage, p: f32) -> Self::Storage;
    fn relu(a: &Self::Storage) -> Self::Storage;
    fn lrelu(a: &Self::Storage, l: f32) -> Self::Storage;
    fn clamp(a: &Self::Storage, min: f32, max: f32) -> Self::Storage;
    fn tanh(a: &Self::Storage) -> Self::Storage;
    fn sigmoid(a: &Self::Storage) -> Self::Storage;
    fn softplus(a: &Self::Storage) -> Self::Storage;

    fn concat(a: &Self::Storage, b: &Self::Storage, axis: usize) -> Self::Storage;
    fn slice(a: &Self::Storage, slice: Vec<SliceAxis>) -> Self::Storage;
    fn reshape(a: &Self::Storage, shape: impl IntoShape) -> Self::Storage;
    fn permute(a: &Self::Storage, axes: &[usize]) -> Self::Storage;
    fn t(a: &Self::Storage) -> Self::Storage;

    fn sum_axis(a: &Self::Storage, axis: usize) -> Self::Storage;
    fn mean_axis(a: &Self::Storage, axis: usize) -> Self::Storage;
    fn max_axis(a: &Self::Storage, axis: usize) -> Self::Storage;
    fn min_axis(a: &Self::Storage, axis: usize) -> Self::Storage;
    fn insert_axis(a: &Self::Storage, axis: usize) -> Self::Storage;
    fn remove_axis(a: &Self::Storage, axis: usize) -> Self::Storage;
    fn broadcast(a: &Self::Storage, shape: impl IntoShape) -> Self::Storage;

    fn scatter_add(a: &mut Self::Storage, slice: Vec<SliceAxis>, b: &Self::Storage);
    fn eq(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
    fn lt(a: &Self::Storage, t: f32) -> Self::Storage;
    fn gt(a: &Self::Storage, t: f32) -> Self::Storage;
    fn lte(a: &Self::Storage, t: f32) -> Self::Storage;
    fn gte(a: &Self::Storage, t: f32) -> Self::Storage;
}

#[derive(Clone)]
pub struct Tensor<B: Backend>(Rc<Inner<B>>);

pub enum Op<B: Backend> {
    Add(Tensor<B>, Tensor<B>),
    Sub(Tensor<B>, Tensor<B>),
    Mul(Tensor<B>, Tensor<B>),
    Div(Tensor<B>, Tensor<B>),
    Matmul(Tensor<B>, Tensor<B>),
    Exp(Tensor<B>),
    Log(Tensor<B>),
    Powf(Tensor<B>, f32),
    Relu(Tensor<B>),
    LRelu(Tensor<B>, f32),
    Clamp(Tensor<B>, f32, f32),
    Tanh(Tensor<B>),
    Sigmoid(Tensor<B>),
    Softplus(Tensor<B>),
    Concat(Tensor<B>, Tensor<B>, usize),
    Slice(Tensor<B>, Vec<SliceAxis>),
    Reshape(Tensor<B>, Vec<usize>),
    Permute(Tensor<B>, Vec<usize>),
    T(Tensor<B>),
    SumAxis(Tensor<B>, usize),
    MeanAxis(Tensor<B>, usize),
    MaxAxis(Tensor<B>, usize),
    MinAxis(Tensor<B>, usize),
    InsertAxis(Tensor<B>, usize),
    RemoveAxis(Tensor<B>, usize),
    Broadcast(Tensor<B>, Vec<usize>),
}

impl<B: Backend> Op<B> {
    pub fn backward(&self, grad: Tensor<B>) {
        match self {
            Op::Add(a, b) | Op::Sub(a, b) => {
                if let Some(a_op) = &a.0.op {
                    a_op.backward(grad.clone());
                }

                if let Some(b_op) = &b.0.op {
                    b_op.backward(grad.clone());
                }
            }
            Op::Mul(a, b) => {
                if let Some(a_op) = &a.0.op {
                    let grad = b * &grad;
                    a_op.backward(grad);
                }

                if let Some(b_op) = &b.0.op {
                    let grad = a * &grad;
                    b_op.backward(grad);
                }
            }
            Op::Div(a, b) => {
                if let Some(a_op) = &a.0.op {
                    let grad = &b.powf(-1.) * &grad;
                    a_op.backward(grad);
                }

                if let Some(b_op) = &b.0.op {
                    let grad = a * &Tensor::scalar(-1.) * &b.powf(-2.) * &grad;
                    b_op.backward(grad);
                }
            }
            Op::Matmul(a, b) => {
                // a: (5, 10) b: (10, 6) -> c: (5, 6)
                // ∂A = c • bT = (5, 6) • (6, 10) -> (5, 10)
                // ∂B = aT • c = (10, 5) • (5, 6) -> (10, 6)

                if let Some(a_op) = &a.0.op {
                    let grad = grad.matmul(&b.t());
                    a_op.backward(grad);
                }

                if let Some(b_op) = &b.0.op {
                    let grad = a.t().matmul(&grad);
                    b_op.backward(grad);
                }
            }
            Op::Exp(a) => {
                if let Some(a_op) = &a.0.op {
                    let grad = a.exp().mul(&grad);
                    a_op.backward(grad);
                }
            }
            Op::Log(a) => {
                if let Some(a_op) = &a.0.op {
                    let grad = (Tensor::scalar(1.) / a) * &grad;
                    a_op.backward(grad);
                }
            }
            Op::Powf(a, pow) => {
                if let Some(a_op) = &a.0.op {
                    let grad = Tensor::scalar(pow - 1.) * a.powf(pow - 1.) * &grad;
                    a_op.backward(grad);
                }
            }
            Op::Relu(a) => {
                if let Some(a_op) = &a.0.op {
                    let grad = a.gt(0.) * grad;
                    a_op.backward(grad);
                }
            }
            Op::LRelu(a, l) => {
                if let Some(a_op) = &a.0.op {
                    let leak = a.lte(0.) * Tensor::scalar(*l);
                    let grad = grad * leak;
                    a_op.backward(grad);
                }
            }
            Op::Clamp(a, min, max) => {
                if let Some(a_op) = &a.0.op {
                    let min = a.gt(*min);
                    let max = a.lt(*max);
                    let clamp = min * max;
                    let grad = clamp * grad;
                    a_op.backward(grad);
                }
            }
            Op::Tanh(a) => {
                if let Some(a_op) = &a.0.op {
                    let tanh = a.tanh();
                    let grad = Tensor::scalar(1.) - tanh.powf(2.);
                    a_op.backward(grad);
                }
            }
            Op::Sigmoid(a) => {
                if let Some(a_op) = &a.0.op {
                    let sigmoid = a.sigmoid();
                    let grad = &sigmoid * (Tensor::scalar(1.) - &sigmoid);
                    a_op.backward(grad);
                }
            }
            Op::Softplus(a) => {
                if let Some(a_op) = &a.0.op {
                    let grad = a.sigmoid();
                    a_op.backward(grad);
                }
            }
            Op::Concat(a, b, axis) => {
                let a_shape = a.shape();
                let mut a_slice = (0..a_shape.len())
                    .map(|_| SliceAxis::Range {
                        start: None,
                        end: None,
                    })
                    .collect::<Vec<_>>();
                let mut b_slice = (0..a_shape.len())
                    .map(|_| SliceAxis::Range {
                        start: None,
                        end: None,
                    })
                    .collect::<Vec<_>>();
                a_slice[*axis] = SliceAxis::Range {
                    start: None,
                    end: Some(a_shape[*axis] as isize),
                };
                b_slice[*axis] = SliceAxis::Range {
                    start: Some(a_shape[*axis] as isize),
                    end: None,
                };

                if let Some(a_op) = &a.0.op {
                    let grad = grad.slice(a_slice);
                    a_op.backward(grad);
                }

                if let Some(b_op) = &b.0.op {
                    let grad = grad.slice(b_slice);
                    b_op.backward(grad)
                }
            }
            Op::Slice(a, slice) => {
                let mut grad_data = B::zeros(a.shape());
                B::scatter_add(&mut grad_data, slice.clone(), &grad.0.data);
                let grad = Tensor::from_data_op(grad_data, None);

                if let Some(a_op) = &a.0.op {
                    a_op.backward(grad);
                }
            }
            Op::Reshape(a, _) => {
                if let Some(a_op) = &a.0.op {
                    let grad = grad.reshape(a.shape());
                    a_op.backward(grad);
                }
            }
            Op::Permute(a, axes) => {
                // a -> [0, 3, 1, 2] -> b
                //
                // [(0, 0), (1, 3), (2, 1), (3, 2)]
                // sort by v.1
                // [(0, 0), (2, 1), (3, 2), (1, 3)]
                // reduce by v.0
                // [0, 2, 3, 1]
                //
                // grad -> [0, 2, 3, 1] -> a
                if let Some(a_op) = &a.0.op {
                    let mut i_perm = axes.iter().cloned().enumerate().collect::<Vec<_>>();
                    i_perm.sort_by(|a, b| a.1.cmp(&b.1));
                    let inv_perm = i_perm.into_iter().map(|e| e.0).collect::<Vec<_>>();

                    let grad = grad.permute(&inv_perm);
                    a_op.backward(grad);
                }
            }
            Op::T(a) => {
                if let Some(a_op) = &a.0.op {
                    let grad = grad.t();
                    a_op.backward(grad);
                }
            }
            Op::SumAxis(a, axis) => {
                if let Some(a_op) = &a.0.op {
                    let grad = grad.insert_axis(*axis).broadcast(a.shape());
                    a_op.backward(grad);
                }
            }
            Op::MeanAxis(a, axis) => {
                if let Some(a_op) = &a.0.op {
                    let d = a.shape()[*axis];
                    let grad =
                        grad.insert_axis(*axis).broadcast(a.shape()) / Tensor::scalar(d as f32);
                    a_op.backward(grad);
                }
            }
            Op::MaxAxis(a, axis) => {
                if let Some(a_op) = &a.0.op {
                    let a_max = a
                        .detach()
                        .max_axis(*axis)
                        .insert_axis(*axis)
                        .broadcast(a.shape());
                    let mask = a.eq(&a_max);
                    let grad = grad.insert_axis(*axis).broadcast(a.shape()) * mask;

                    a_op.backward(grad);
                }
            }
            Op::MinAxis(a, axis) => {
                if let Some(a_op) = &a.0.op {
                    let a_min = a
                        .detach()
                        .min_axis(*axis)
                        .insert_axis(*axis)
                        .broadcast(a.shape());
                    let mask = a.eq(&a_min);
                    let grad = grad.insert_axis(*axis).broadcast(a.shape()) * mask;

                    a_op.backward(grad);
                }
            }
            Op::InsertAxis(a, axis) => {
                if let Some(a_op) = &a.0.op {
                    a_op.backward(grad.remove_axis(*axis));
                }
            }
            Op::RemoveAxis(a, axis) => {
                if let Some(a_op) = &a.0.op {
                    a_op.backward(grad.insert_axis(*axis));
                }
            }
            Op::Broadcast(a, _) => {
                if let Some(a_op) = &a.0.op {
                    // ranks should be equal before broadcast, enforced.

                    let mut grad = grad;
                    for axis in (0..a.shape().len()).rev() {
                        if a.shape()[axis] == 1 && grad.shape()[axis] != 1 {
                            grad = grad.sum_axis(axis);
                        }
                    }

                    a_op.backward(grad);
                }
            }
        }
    }
}

pub struct Inner<B: Backend> {
    pub id: usize,
    pub data: B::Storage,
    pub grad: Option<RefCell<B::Storage>>,
    pub op: Option<Op<B>>,
    pub graph: bool,
}

impl<B: Backend> Tensor<B> {
    fn from_data_op(data: B::Storage, op: Option<Op<B>>) -> Self {
        Self(Rc::new(Inner {
            id: 0,
            data,
            grad: None,
            op,
            graph: true,
        }))
    }

    pub fn detach(&self) -> Self {
        Self(Rc::new(Inner {
            id: 0,
            data: self.0.data.clone(),
            grad: None,
            op: None,
            graph: false,
        }))
    }

    pub fn scalar(x: f32) -> Tensor<B> {
        Self::from_data_op(B::scalar(x), None)
    }

    pub fn zeros(shape: impl IntoShape) -> Tensor<B> {
        Self::from_data_op(B::zeros(shape), None)
    }

    pub fn ones(shape: impl IntoShape) -> Tensor<B> {
        Self::from_data_op(B::ones(shape), None)
    }

    pub fn randn(shape: impl IntoShape) -> Tensor<B> {
        Self::from_data_op(B::randn(shape), None)
    }

    pub fn shape(&self) -> &[usize] {
        B::shape(&self.0.data)
    }

    pub fn add(&self, rhs: &Tensor<B>) -> Tensor<B> {
        let op = (self.0.graph || rhs.0.graph).then_some(Op::Add(self.clone(), rhs.clone()));
        Self::from_data_op(B::add(&self.0.data, &rhs.0.data), op)
    }

    pub fn sub(&self, rhs: &Tensor<B>) -> Tensor<B> {
        let op = (self.0.graph || rhs.0.graph).then_some(Op::Sub(self.clone(), rhs.clone()));
        Self::from_data_op(B::sub(&self.0.data, &rhs.0.data), op)
    }

    pub fn mul(&self, rhs: &Tensor<B>) -> Tensor<B> {
        let op = (self.0.graph || rhs.0.graph).then_some(Op::Mul(self.clone(), rhs.clone()));
        Self::from_data_op(B::mul(&self.0.data, &rhs.0.data), op)
    }

    pub fn div(&self, rhs: &Tensor<B>) -> Tensor<B> {
        let op = (self.0.graph || rhs.0.graph).then_some(Op::Div(self.clone(), rhs.clone()));
        Self::from_data_op(B::div(&self.0.data, &rhs.0.data), op)
    }

    pub fn matmul(&self, rhs: &Tensor<B>) -> Tensor<B> {
        let op = (self.0.graph || rhs.0.graph).then_some(Op::Matmul(self.clone(), rhs.clone()));
        Self::from_data_op(B::matmul(&self.0.data, &rhs.0.data), op)
    }

    pub fn exp(&self) -> Tensor<B> {
        Self::from_data_op(
            B::exp(&self.0.data),
            self.0.graph.then_some(Op::Exp(self.clone())),
        )
    }

    pub fn log(&self) -> Tensor<B> {
        Self::from_data_op(
            B::log(&self.0.data),
            self.0.graph.then_some(Op::Log(self.clone())),
        )
    }

    pub fn powf(&self, power: f32) -> Tensor<B> {
        Self::from_data_op(
            B::powf(&self.0.data, power),
            self.0.graph.then_some(Op::Powf(self.clone(), power)),
        )
    }

    pub fn relu(&self) -> Tensor<B> {
        Self::from_data_op(
            B::relu(&self.0.data),
            self.0.graph.then_some(Op::Relu(self.clone())),
        )
    }

    pub fn lrelu(&self, l: f32) -> Tensor<B> {
        Self::from_data_op(
            B::lrelu(&self.0.data, l),
            self.0.graph.then_some(Op::LRelu(self.clone(), l)),
        )
    }

    pub fn clamp(&self, min: f32, max: f32) -> Tensor<B> {
        Self::from_data_op(
            B::clamp(&self.0.data, min, max),
            self.0.graph.then_some(Op::Clamp(self.clone(), min, max)),
        )
    }

    pub fn tanh(&self) -> Tensor<B> {
        Self::from_data_op(
            B::tanh(&self.0.data),
            self.0.graph.then_some(Op::Tanh(self.clone())),
        )
    }

    pub fn sigmoid(&self) -> Tensor<B> {
        Self::from_data_op(
            B::sigmoid(&self.0.data),
            self.0.graph.then_some(Op::Sigmoid(self.clone())),
        )
    }

    pub fn softplus(&self) -> Tensor<B> {
        Self::from_data_op(
            B::softplus(&self.0.data),
            self.0.graph.then_some(Op::Softplus(self.clone())),
        )
    }

    pub fn concat(&self, rhs: &Tensor<B>, axis: usize) -> Tensor<B> {
        let op =
            (self.0.graph || rhs.0.graph).then_some(Op::Concat(self.clone(), rhs.clone(), axis));
        Self::from_data_op(B::concat(&self.0.data, &rhs.0.data, axis), op)
    }

    pub fn slice(&self, slice: Vec<SliceAxis>) -> Tensor<B> {
        Self::from_data_op(
            B::slice(&self.0.data, slice.clone()),
            self.0.graph.then_some(Op::Slice(self.clone(), slice)),
        )
    }

    pub fn reshape(&self, shape: impl IntoShape) -> Tensor<B> {
        let shape = shape.into_shape();
        Self::from_data_op(
            B::reshape(&self.0.data, shape.clone()),
            self.0.graph.then_some(Op::Reshape(self.clone(), shape)),
        )
    }

    pub fn permute(&self, axes: &[usize]) -> Tensor<B> {
        Self::from_data_op(
            B::permute(&self.0.data, axes),
            self.0
                .graph
                .then_some(Op::Permute(self.clone(), axes.to_vec())),
        )
    }

    pub fn t(&self) -> Tensor<B> {
        Self::from_data_op(
            B::t(&self.0.data),
            self.0.graph.then_some(Op::T(self.clone())),
        )
    }

    pub fn sum_axis(&self, axis: usize) -> Tensor<B> {
        Self::from_data_op(
            B::sum_axis(&self.0.data, axis),
            self.0.graph.then_some(Op::SumAxis(self.clone(), axis)),
        )
    }

    pub fn mean_axis(&self, axis: usize) -> Tensor<B> {
        Self::from_data_op(
            B::mean_axis(&self.0.data, axis),
            self.0.graph.then_some(Op::MeanAxis(self.clone(), axis)),
        )
    }

    pub fn max_axis(&self, axis: usize) -> Tensor<B> {
        Self::from_data_op(
            B::max_axis(&self.0.data, axis),
            self.0.graph.then_some(Op::MaxAxis(self.clone(), axis)),
        )
    }

    pub fn min_axis(&self, axis: usize) -> Tensor<B> {
        Self::from_data_op(
            B::min_axis(&self.0.data, axis),
            self.0.graph.then_some(Op::MinAxis(self.clone(), axis)),
        )
    }

    pub fn insert_axis(&self, axis: usize) -> Tensor<B> {
        Self::from_data_op(
            B::insert_axis(&self.0.data, axis),
            self.0.graph.then_some(Op::InsertAxis(self.clone(), axis)),
        )
    }

    pub fn remove_axis(&self, axis: usize) -> Tensor<B> {
        Self::from_data_op(
            B::remove_axis(&self.0.data, axis),
            self.0.graph.then_some(Op::RemoveAxis(self.clone(), axis)),
        )
    }

    pub fn broadcast(&self, shape: impl IntoShape) -> Tensor<B> {
        let shape = shape.into_shape();
        Self::from_data_op(
            B::broadcast(&self.0.data, shape.clone()),
            self.0.graph.then_some(Op::Broadcast(self.clone(), shape)),
        )
    }

    // equality

    pub fn eq(&self, rhs: &Tensor<B>) -> Tensor<B> {
        Self::from_data_op(B::eq(&self.0.data, &rhs.0.data), None)
    }

    pub fn lt(&self, t: f32) -> Tensor<B> {
        Self::from_data_op(B::lt(&self.0.data, t), None)
    }

    pub fn gt(&self, t: f32) -> Tensor<B> {
        Self::from_data_op(B::gt(&self.0.data, t), None)
    }

    pub fn lte(&self, t: f32) -> Tensor<B> {
        Self::from_data_op(B::lte(&self.0.data, t), None)
    }

    pub fn gte(&self, t: f32) -> Tensor<B> {
        Self::from_data_op(B::gte(&self.0.data, t), None)
    }

    // diff

    pub fn backward(&self) {
        debug_assert!(
            self.shape().len() > 0,
            "cannot call backward() on a > 0 dim tensor"
        );
        debug_assert!(matches!(&self.0.op, Some(_)), "this tensor has no graph");

        let seed = Tensor::<B>::scalar(1.);
        if let Some(op) = &self.0.op {
            op.backward(seed);
        }
    }
}

pub trait IntoShape {
    fn into_shape(self) -> Vec<usize>;
}

impl IntoShape for usize {
    fn into_shape(self) -> Vec<usize> {
        vec![self]
    }
}

impl IntoShape for (usize,) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0]
    }
}

impl IntoShape for (usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1]
    }
}

impl IntoShape for (usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2]
    }
}

impl IntoShape for (usize, usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3]
    }
}

impl IntoShape for (usize, usize, usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3, self.4]
    }
}

impl IntoShape for Vec<usize> {
    fn into_shape(self) -> Vec<usize> {
        self
    }
}

impl IntoShape for &[usize] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

#[derive(Clone)]
pub enum SliceAxis {
    Index(isize),
    Range {
        start: Option<isize>,
        end: Option<isize>,
    },
}

#[macro_export]
macro_rules! slice {
    (@convert $r:expr) => {
        $crate::tensor::SliceAxis::from($r)
    };
    // last item
    (@parse [$($acc:expr),*] $r:expr) => {
        &[$($acc,)* s!(@convert $r)]
    };
    // more items
    (@parse [$($acc:expr),*] $r:expr, $($rest:tt)*) => {
        s!(@parse [$($acc,)* s!(@convert $r)] $($rest)*)
    };
    // entry
    ($($t:tt)*) => {
        s!(@parse [] $($t)*)
    };
}

macro_rules! impl_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<B: Backend> $trait<&Tensor<B>> for &Tensor<B> {
            type Output = Tensor<B>;
            fn $method(self, rhs: &Tensor<B>) -> Tensor<B> {
                Tensor::<B>::$method(&self, rhs)
            }
        }

        impl<B: Backend> $trait<&Tensor<B>> for Tensor<B> {
            type Output = Tensor<B>;
            fn $method(self, rhs: &Tensor<B>) -> Tensor<B> {
                Tensor::<B>::$method(&self, rhs)
            }
        }

        impl<B: Backend> $trait<Tensor<B>> for Tensor<B> {
            type Output = Tensor<B>;
            fn $method(self, rhs: Tensor<B>) -> Tensor<B> {
                Tensor::<B>::$method(&self, &rhs)
            }
        }

        impl<B: Backend> $trait<Tensor<B>> for &Tensor<B> {
            type Output = Tensor<B>;
            fn $method(self, rhs: Tensor<B>) -> Tensor<B> {
                Tensor::<B>::$method(self, &rhs)
            }
        }
    };
}

impl_op!(Add, add, +);
impl_op!(Sub, sub, -);
impl_op!(Mul, mul, *);
impl_op!(Div, div, /);
