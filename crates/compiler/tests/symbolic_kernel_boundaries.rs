use bml_compiler::{
    symbolic_kernels::{compile_operation, KernelLimits, Operation},
    RpnOp,
};

#[test]
fn rms_signature_rejects_invalid_epsilon_before_exposing_configuration() {
    for epsilon in [0.0, -0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert!(
            Operation::Norm.signature(4, epsilon).is_err(),
            "epsilon={epsilon}"
        );
        assert!(compile_operation(Operation::Norm, 4, epsilon, KernelLimits::default()).is_err());
    }
    let signature = Operation::Norm.signature(4, 0.5).unwrap();
    assert_eq!(
        (signature.inputs, signature.external, signature.outputs),
        (4, 4, 4)
    );
    assert_eq!(signature.fixed, [0.25, 0.5, 0.5]);
    assert!(compile_operation(Operation::Norm, 4, 0.5, KernelLimits::default()).is_ok());
}

#[test]
fn fixed_width_signatures_reject_other_nonempty_widths() {
    for (operation, accepted) in [
        (Operation::Rope, 2),
        (Operation::Swiglu, 1),
        (Operation::Add, 1),
        (Operation::Scale, 1),
    ] {
        for width in [0, 1, 2, 3, usize::MAX] {
            assert_eq!(
                operation.signature(width, 0.5).is_ok(),
                width == accepted,
                "{operation:?} width={width}"
            );
        }
    }
}

#[test]
fn fragment_budget_must_fit_one_opcode_and_accepts_the_exact_boundary() {
    let size = std::mem::size_of::<RpnOp>();
    for fragment_bytes in [0, size - 1] {
        assert!(compile_operation(
            Operation::Add,
            1,
            0.5,
            KernelLimits {
                fragment_bytes,
                ..Default::default()
            },
        )
        .is_err());
    }
    let kernel = compile_operation(
        Operation::Add,
        1,
        0.5,
        KernelLimits {
            fragment_bytes: size,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(kernel.graphs.len(), 1);
    assert!(kernel.graphs[0].fragments.len() > 1);
    for fragment in &kernel.graphs[0].fragments {
        assert_eq!(fragment.ops.len(), 1);
        assert_eq!(fragment.byte_size(), size);
    }
}
