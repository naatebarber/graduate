use ndarray::{ArrayD, Axis, Ix2, SliceInfoElem, Zip};
use ndarray_rand::{RandomExt, rand_distr::Normal};

use crate::{Backend, SliceAxis};

#[derive(Clone)]
pub struct NDArrayBackend {}

impl NDArrayBackend {
    fn slice_axis_to_ndarray(slice: &[SliceAxis]) -> Vec<SliceInfoElem> {
        slice
            .into_iter()
            .map(|slice| match slice {
                SliceAxis::Index(i) => SliceInfoElem::Index(*i),
                SliceAxis::Range { start, end } => SliceInfoElem::Slice {
                    start: start.unwrap_or(0),
                    end: *end,
                    step: 1,
                },
            })
            .collect::<Vec<SliceInfoElem>>()
    }
}

impl Backend for NDArrayBackend {
    type Storage = ArrayD<f32>;

    fn scalar(x: f32) -> Self::Storage {
        ArrayD::from_elem(vec![], x)
    }

    fn zeros(shape: impl crate::IntoShape) -> Self::Storage {
        ArrayD::zeros(shape.into_shape())
    }

    fn ones(shape: impl crate::IntoShape) -> Self::Storage {
        ArrayD::ones(shape.into_shape())
    }

    fn randn(shape: impl crate::IntoShape) -> Self::Storage {
        ArrayD::random(shape.into_shape(), Normal::new(0., 1.).unwrap())
    }

    fn shape(a: &Self::Storage) -> &[usize] {
        a.shape()
    }

    fn add(a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        a + b
    }

    fn sub(a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        a - b
    }

    fn mul(a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        a * b
    }

    fn div(a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        a / b
    }

    fn matmul(a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        debug_assert!(
            a.shape().len() == 2 && b.shape().len() == 2,
            "cannot matmul rank {} tensor with rank {} tensor",
            a.shape().len(),
            b.shape().len()
        );
        let a_2 = a.view().into_dimensionality::<Ix2>().unwrap();
        let b_2 = b.view().into_dimensionality::<Ix2>().unwrap();
        a_2.dot(&b_2).into_dyn()
    }

    fn exp(a: &Self::Storage) -> Self::Storage {
        a.exp()
    }

    fn log(a: &Self::Storage) -> Self::Storage {
        a.ln()
    }

    fn powf(a: &Self::Storage, p: f32) -> Self::Storage {
        a.powf(p)
    }

    fn relu(a: &Self::Storage) -> Self::Storage {
        a.mapv(|x| x.max(0.))
    }

    fn lrelu(a: &Self::Storage, l: f32) -> Self::Storage {
        a.mapv(|x| if x > 0. { x } else { l * x })
    }

    fn clamp(a: &Self::Storage, min: f32, max: f32) -> Self::Storage {
        a.clamp(min, max)
    }

    fn tanh(a: &Self::Storage) -> Self::Storage {
        a.tanh()
    }

    fn sigmoid(a: &Self::Storage) -> Self::Storage {
        1. / (1. + (-a).exp())
    }

    fn softplus(a: &Self::Storage) -> Self::Storage {
        (1. + a.exp()).ln()
    }

    fn concat(a: &Self::Storage, b: &Self::Storage, axis: usize) -> Self::Storage {
        ndarray::concatenate(Axis(axis), &[a.view(), b.view()]).unwrap()
    }

    fn slice(a: &Self::Storage, slice: Vec<crate::SliceAxis>) -> Self::Storage {
        let slice = NDArrayBackend::slice_axis_to_ndarray(&slice);
        a.slice(slice.as_slice()).to_owned()
    }

    fn reshape(a: &Self::Storage, shape: impl crate::IntoShape) -> Self::Storage {
        a.clone().into_shape_clone(shape.into_shape()).unwrap()
    }

    fn permute(a: &Self::Storage, axes: &[usize]) -> Self::Storage {
        a.clone().permuted_axes(axes)
    }

    fn t(a: &Self::Storage) -> Self::Storage {
        a.view().t().to_owned()
    }

    fn sum_axis(a: &Self::Storage, axis: usize) -> Self::Storage {
        a.sum_axis(Axis(axis))
    }

    fn mean_axis(a: &Self::Storage, axis: usize) -> Self::Storage {
        a.mean_axis(Axis(axis)).unwrap()
    }

    /// Not implemented yet
    fn max_axis(a: &Self::Storage, axis: usize) -> Self::Storage {
        a.clone()
    }

    /// Not implemented yet
    fn min_axis(a: &Self::Storage, axis: usize) -> Self::Storage {
        a.clone()
    }

    fn insert_axis(a: &Self::Storage, axis: usize) -> Self::Storage {
        a.clone().insert_axis(Axis(axis))
    }

    fn remove_axis(a: &Self::Storage, axis: usize) -> Self::Storage {
        a.clone().remove_axis(Axis(axis))
    }

    fn broadcast(a: &Self::Storage, shape: impl crate::IntoShape) -> Self::Storage {
        a.broadcast(shape.into_shape()).unwrap().to_owned()
    }

    fn scatter_add(a: &mut Self::Storage, slice: Vec<SliceAxis>, b: &Self::Storage) {
        let slice = NDArrayBackend::slice_axis_to_ndarray(&slice);
        let mut region = a.slice_mut(slice.as_slice());
        region += b;
    }

    fn eq(a: &Self::Storage, b: &Self::Storage) -> Self::Storage {
        let mut mask = ArrayD::zeros(a.shape());
        Zip::from(&mut mask).and(a).and(b).for_each(|mask, a, b| {
            if *a == *b {
                *mask = 1.
            }
        });

        mask
    }

    fn lt(a: &Self::Storage, t: f32) -> Self::Storage {
        a.mapv(|v| if v < t { 1. } else { 0. })
    }

    fn gt(a: &Self::Storage, t: f32) -> Self::Storage {
        a.mapv(|v| if v > t { 1. } else { 0. })
    }

    fn lte(a: &Self::Storage, t: f32) -> Self::Storage {
        a.mapv(|v| if v < t { 1. } else { 0. })
    }

    fn gte(a: &Self::Storage, t: f32) -> Self::Storage {
        a.mapv(|v| if v > t { 1. } else { 0. })
    }
}
