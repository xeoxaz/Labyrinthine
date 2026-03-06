use std::process::Command;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuPolicy {
    pub prefer_gpu: bool,
    pub require_gpu: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuProbe {
    pub cuda: bool,
    pub rocm: bool,
    pub vulkan: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlBackend {
    Cpu,
    CudaReady,
    RocmReady,
    VulkanReady,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlRuntime {
    backend: MlBackend,
    detail_message: String,
}

pub fn detect_gpu_probe() -> GpuProbe {
    GpuProbe {
        cuda: command_succeeds("nvidia-smi", &["-L"]),
        rocm: command_succeeds("rocminfo", &[]),
        vulkan: command_succeeds("vulkaninfo", &["--summary"]),
    }
}

fn command_succeeds(program: &str, args: &[&str]) -> bool {
    Command::new(program)
        .args(args)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

pub fn resolve_ml_runtime(policy: GpuPolicy, probe: GpuProbe) -> Result<MlRuntime, String> {
    let backend = if policy.prefer_gpu {
        if probe.cuda {
            Some(MlBackend::CudaReady)
        } else if probe.rocm {
            Some(MlBackend::RocmReady)
        } else if probe.vulkan {
            Some(MlBackend::VulkanReady)
        } else {
            None
        }
    } else {
        None
    };

    if policy.require_gpu && backend.is_none() {
        return Err(
            "GPU mode required, but no supported backend was detected (checked CUDA, ROCm, Vulkan)."
                .to_string(),
        );
    }

    Ok(match backend {
        Some(backend) => MlRuntime {
            backend,
            detail_message: format!("ml {} ready", backend.short_label().to_ascii_lowercase()),
        },
        None if policy.prefer_gpu => MlRuntime {
            backend: MlBackend::Cpu,
            detail_message: "ml cpu fallback".to_string(),
        },
        None => MlRuntime {
            backend: MlBackend::Cpu,
            detail_message: "ml cpu-only".to_string(),
        },
    })
}

impl MlRuntime {
    pub fn short_label(&self) -> &'static str {
        self.backend.short_label()
    }

    pub fn detail_message(&self) -> &str {
        &self.detail_message
    }
}

impl MlBackend {
    pub fn short_label(self) -> &'static str {
        match self {
            MlBackend::Cpu => "CPU",
            MlBackend::CudaReady => "CUDA",
            MlBackend::RocmReady => "ROCM",
            MlBackend::VulkanReady => "VK",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_cuda_when_available() {
        let runtime = resolve_ml_runtime(
            GpuPolicy {
                prefer_gpu: true,
                require_gpu: false,
            },
            GpuProbe {
                cuda: true,
                rocm: true,
                vulkan: true,
            },
        )
        .unwrap();

        assert_eq!(runtime.short_label(), "CUDA");
        assert_ne!(runtime.backend, MlBackend::Cpu);
    }

    #[test]
    fn falls_back_to_cpu_when_gpu_missing() {
        let runtime = resolve_ml_runtime(
            GpuPolicy {
                prefer_gpu: true,
                require_gpu: false,
            },
            GpuProbe::default(),
        )
        .unwrap();

        assert_eq!(runtime.short_label(), "CPU");
        assert_eq!(runtime.detail_message(), "ml cpu fallback");
    }

    #[test]
    fn require_gpu_errors_without_backend() {
        let result = resolve_ml_runtime(
            GpuPolicy {
                prefer_gpu: true,
                require_gpu: true,
            },
            GpuProbe::default(),
        );

        assert!(result.is_err());
    }
}