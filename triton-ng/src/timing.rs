pub(crate) fn monotonic_nanos() -> u64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    assert_eq!(
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) },
        0,
        "CLOCK_MONOTONIC is unavailable"
    );
    (time.tv_sec as u64) * 1_000_000_000 + (time.tv_nsec as u64)
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
    last_response_success: Option<bool>,
}

impl RequestTimestamps {
    pub(crate) fn new() -> Self {
        Self {
            exec_start: monotonic_nanos(),
            compute_start: None,
            compute_end: None,
            exec_end: None,
            last_response: None,
            last_response_success: None,
        }
    }

    pub(crate) fn start_compute(&mut self, at: u64) {
        self.compute_start.get_or_insert(at);
    }

    pub(crate) fn record_response(&mut self, response: ResponseTimestamps, success: bool) {
        debug_assert!(response.response_start >= self.exec_start);
        debug_assert!(response.response_end >= response.response_start);
        self.compute_end = response.compute_output_start;
        self.exec_end = Some(response.response_end);
        self.last_response = Some(response);
        self.last_response_success = Some(success);
    }

    pub(crate) fn success(&self, empty_success: bool) -> bool {
        self.last_response_success.unwrap_or(empty_success)
    }

    pub(crate) fn statistics(&self, at: u64) -> [u64; 4] {
        let compute_start = self.compute_start.unwrap_or(self.exec_start);
        let compute_end = self
            .compute_end
            .or_else(|| self.last_response.map(|response| response.response_end))
            .unwrap_or(at);
        [
            self.exec_start,
            compute_start,
            compute_end,
            self.exec_end.unwrap_or(at),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PendingResponseTimestamps, RequestTimestamps, ResponseTimestamps, monotonic_nanos,
    };

    fn clock_monotonic_nanos() -> u64 {
        let mut time = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        assert_eq!(
            unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) },
            0
        );
        (time.tv_sec as u64) * 1_000_000_000 + (time.tv_nsec as u64)
    }

    #[test]
    fn monotonic_nanos_uses_the_same_epoch_as_triton() {
        let before = clock_monotonic_nanos();
        let timestamp = monotonic_nanos();
        let after = clock_monotonic_nanos();
        assert!(before <= timestamp && timestamp <= after);
    }

    #[test]
    fn compute_start_is_recorded_only_once() {
        let mut timestamps = RequestTimestamps::new();
        timestamps.start_compute(10);
        timestamps.start_compute(20);
        assert_eq!(timestamps.compute_start, Some(10));
    }

    #[test]
    fn last_response_determines_request_success() {
        let mut timestamps = RequestTimestamps::new();
        timestamps.exec_start = 0;
        assert!(!timestamps.success(false));
        assert!(timestamps.success(true));

        let response = ResponseTimestamps {
            response_start: 10,
            compute_output_start: Some(20),
            response_end: 30,
        };
        timestamps.record_response(response, true);
        assert!(timestamps.success(false));
        timestamps.record_response(response, false);
        assert!(!timestamps.success(true));
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
        timestamps.record_response(first, true);
        timestamps.record_response(last, true);

        assert_eq!(timestamps.compute_end, Some(50));
        assert_eq!(timestamps.exec_end, Some(60));
        assert_eq!(timestamps.last_response.unwrap().response_start, 40);
    }

    #[test]
    fn request_statistics_fill_missing_computation_points() {
        let mut timestamps = RequestTimestamps::new();
        timestamps.exec_start = 10;
        assert_eq!(timestamps.statistics(40), [10, 10, 40, 40]);

        timestamps.start_compute(20);
        timestamps.record_response(
            ResponseTimestamps {
                response_start: 30,
                compute_output_start: None,
                response_end: 35,
            },
            false,
        );
        assert_eq!(timestamps.statistics(50), [10, 20, 35, 35]);
    }

    #[test]
    fn response_without_outputs_has_no_compute_output_start() {
        let response = PendingResponseTimestamps::new().finish(20);
        assert_eq!(response.compute_output_start, None);
        assert_eq!(response.response_end, 20);
    }
}
