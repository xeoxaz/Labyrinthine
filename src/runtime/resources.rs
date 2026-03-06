use rayon::ThreadPoolBuilder;

#[derive(Clone, Copy, Debug)]
pub struct ResourcePolicy {
    pub max_mode: bool,
    pub threads: Option<usize>,
}

pub fn resolve_thread_count(policy: ResourcePolicy) -> usize {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    if let Some(explicit) = policy.threads {
        explicit.max(1)
    } else if policy.max_mode {
        available
    } else {
        available.saturating_sub(1).max(1)
    }
}

pub fn apply(policy: ResourcePolicy) -> Result<(), String> {
    let threads = resolve_thread_count(policy);

    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|err| format!("failed to build rayon global pool: {err}"))
}
