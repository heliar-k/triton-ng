use std::sync::OnceLock;
use std::time::Instant;

pub(crate) fn monotonic_nanos() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    let nanos = START.get_or_init(Instant::now).elapsed().as_nanos();
    u64::try_from(nanos).unwrap_or(u64::MAX)
}

#[derive(Clone, Copy)]
pub(crate) struct ResponseTimestamps {
    pub(crate) response_start: u64,
    pub(crate) compute_output_start: Option<u64>,
    pub(crate) response_end: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct PendingResponseTimestamps {
    pub(crate) response_start: u64,
    pub(crate) compute_output_start: Option<u64>,
}

impl PendingResponseTimestamps {
    pub(crate) fn new() -> Self {
        Self {
            response_start: monotonic_nanos(),
            compute_output_start: None,
        }
    }

    pub(crate) fn finish(self, response_end: u64) -> ResponseTimestamps {
        ResponseTimestamps {
            response_start: self.response_start,
            compute_output_start: self.compute_output_start,
            response_end,
        }
    }
}

pub(crate) struct RequestTimestamps {
    pub(crate) exec_start: u64,
    pub(crate) compute_start: Option<u64>,
    pub(crate) compute_end: Option<u64>,
    pub(crate) exec_end: Option<u64>,
    pub(crate) last_response: Option<ResponseTimestamps>,
}

impl RequestTimestamps {
    pub(crate) fn new() -> Self {
        Self {
            exec_start: monotonic_nanos(),
            compute_start: None,
            compute_end: None,
            exec_end: None,
            last_response: None,
        }
    }

    pub(crate) fn start_compute(&mut self, at: u64) {
        self.compute_start.get_or_insert(at);
    }

    pub(crate) fn record_response(&mut self, response: ResponseTimestamps) {
        debug_assert!(response.response_start >= self.exec_start);
        debug_assert!(response.response_end >= response.response_start);
        self.compute_end = response.compute_output_start;
        self.exec_end = Some(response.response_end);
        self.last_response = Some(response);
    }
}

#[cfg(test)]
mod tests {
    use super::{PendingResponseTimestamps, RequestTimestamps, ResponseTimestamps};

    #[test]
    fn compute_start_is_recorded_only_once() {
        let mut timestamps = RequestTimestamps::new();
        timestamps.start_compute(10);
        timestamps.start_compute(20);
        assert_eq!(timestamps.compute_start, Some(10));
    }

    #[test]
    fn last_response_supplies_request_end_timestamps() {
        let mut timestamps = RequestTimestamps::new();
        timestamps.exec_start = 0;
        let first = ResponseTimestamps {
            response_start: 10,
            compute_output_start: Some(20),
            response_end: 30,
        };
        let last = ResponseTimestamps {
            response_start: 40,
            compute_output_start: Some(50),
            response_end: 60,
        };
        timestamps.record_response(first);
        timestamps.record_response(last);

        assert_eq!(timestamps.compute_end, Some(50));
        assert_eq!(timestamps.exec_end, Some(60));
        assert_eq!(timestamps.last_response.unwrap().response_start, 40);
    }

    #[test]
    fn response_without_outputs_has_no_compute_output_start() {
        let response = PendingResponseTimestamps::new().finish(20);
        assert_eq!(response.compute_output_start, None);
        assert_eq!(response.response_end, 20);
    }
}
