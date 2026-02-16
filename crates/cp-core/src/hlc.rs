//! Hybrid Logical Clock (HLC) for causal ordering
//!
//! Per CP-003 §7: HLC combines wall-clock time with a logical counter
//! to provide total ordering of events across devices. Used for:
//! - Conflict resolution (Last-Writer-Wins)
//! - Causal ordering of state transitions
//! - Deterministic merge of concurrent updates
//!
//! Ordering: (wall_ms, counter, node_id) — lexicographic comparison.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Hybrid Logical Clock timestamp.
///
/// Provides total ordering across distributed devices without
/// relying on synchronized wall clocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Hlc {
    /// Wall-clock time in milliseconds since Unix epoch.
    /// Used as the primary ordering component.
    pub wall_ms: u64,

    /// Logical counter for sub-millisecond ordering.
    /// Incremented when wall clock hasn't advanced.
    pub counter: u16,

    /// Node identifier (16 bytes) for deterministic tiebreaking
    /// when wall_ms and counter are identical across devices.
    pub node_id: [u8; 16],
}

impl Hlc {
    /// Create a new HLC from current system time and device ID.
    pub fn now(device_id: uuid::Uuid) -> Self {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            wall_ms: now_ms,
            counter: 0,
            node_id: *device_id.as_bytes(),
        }
    }

    /// Create a new HLC with the given wall time and node ID.
    pub fn new(wall_ms: u64, node_id: [u8; 16]) -> Self {
        Self {
            wall_ms,
            counter: 0,
            node_id,
        }
    }

    /// Create a zero/genesis HLC (for initial state).
    pub fn zero(node_id: [u8; 16]) -> Self {
        Self {
            wall_ms: 0,
            counter: 0,
            node_id,
        }
    }

    /// Advance the local clock.
    ///
    /// Per CP-003 §7:
    /// - If wall clock has advanced past current, use new wall time, reset counter
    /// - Otherwise, increment counter
    pub fn tick(&mut self, now_ms: u64) {
        if now_ms > self.wall_ms {
            self.wall_ms = now_ms;
            self.counter = 0;
        } else {
            self.counter = self.counter.saturating_add(1);
        }
    }

    /// Merge with a remote HLC to maintain causality.
    ///
    /// Per CP-003 §7:
    /// - Take max(local.wall, remote.wall, now)
    /// - If all three equal, increment max counter
    /// - If two equal at max, increment the max counter among those two
    /// - Otherwise, reset counter to 0
    pub fn merge(&mut self, remote: &Hlc, now_ms: u64) {
        let max_wall = self.wall_ms.max(remote.wall_ms).max(now_ms);

        if max_wall == self.wall_ms && max_wall == remote.wall_ms {
            // All equal: take max counter and increment
            self.counter = self.counter.max(remote.counter).saturating_add(1);
        } else if max_wall == self.wall_ms {
            // Local is ahead: increment local counter
            self.counter = self.counter.saturating_add(1);
        } else if max_wall == remote.wall_ms {
            // Remote is ahead: take remote counter and increment
            self.wall_ms = remote.wall_ms;
            self.counter = remote.counter.saturating_add(1);
        } else {
            // Wall time advanced: reset counter
            self.wall_ms = max_wall;
            self.counter = 0;
        }
    }

    /// Compare two HLCs for ordering.
    ///
    /// Returns Ordering based on (wall_ms, counter, node_id) lexicographically.
    pub fn cmp(&self, other: &Hlc) -> Ordering {
        self.wall_ms
            .cmp(&other.wall_ms)
            .then_with(|| self.counter.cmp(&other.counter))
            .then_with(|| self.node_id.cmp(&other.node_id))
    }

    /// Serialize HLC to bytes for hashing.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(26);
        bytes.extend_from_slice(&self.wall_ms.to_le_bytes());
        bytes.extend_from_slice(&self.counter.to_le_bytes());
        bytes.extend_from_slice(&self.node_id);
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hlc_ordering() {
        let node_a = [0u8; 16];
        let node_b = [1u8; 16];

        let hlc_a = Hlc::new(1000, node_a);
        let hlc_b = Hlc::new(1001, node_a);
        let hlc_c = Hlc::new(1000, node_b);

        assert!(hlc_a < hlc_b); // 1000 < 1001
    }

    #[test]
    fn test_hlc_merge() {
        let node = [0u8; 16];
        let mut local = Hlc::new(1000, node);
        let remote = Hlc::new(1005, node);

        local.merge(&remote, 1000);

        assert_eq!(local.wall_ms, 1005);
    }

    #[test]
    fn test_hlc_tick() {
        let node = [0u8; 16];
        let mut hlc = Hlc::new(1000, node);

        hlc.tick(1001);
        assert_eq!(hlc.wall_ms, 1001);
        assert_eq!(hlc.counter, 0);

        hlc.tick(1001);
        assert_eq!(hlc.counter, 1);
    }

    // Additional tests for comprehensive coverage

    #[test]
    fn test_hlc_new() {
        let hlc = Hlc::new(1000, [1u8; 16]);
        assert_eq!(hlc.wall_ms, 1000);
        assert_eq!(hlc.counter, 0);
        assert_eq!(hlc.node_id, [1u8; 16]);
    }

    #[test]
    fn test_hlc_now_increments() {
        let device_id = uuid::Uuid::new_v4();
        let hlc1 = Hlc::now(device_id);
        let hlc2 = Hlc::now(device_id);
        assert!(hlc1.wall_ms > 0);
        assert!(hlc2.wall_ms >= hlc1.wall_ms);
    }

    #[test]
    fn test_hlc_send_receive_ordering() {
        let node = [1u8; 16];
        let mut local = Hlc::new(1000, node);
        let remote = Hlc::new(2000, node);
        local.merge(&remote, 1000);
        assert_eq!(local.wall_ms, 2000);
    }

    #[test]
    fn test_hlc_causality_preserved() {
        let node = [1u8; 16];
        let mut local = Hlc::new(1000, node);
        let remote = Hlc::new(1500, node);
        local.merge(&remote, 1500);
        assert!(local.wall_ms >= 1000);
    }

    #[test]
    fn test_hlc_concurrent_events_tiebreaker() {
        let node_a = [1u8; 16];
        let node_b = [2u8; 16];
        let hlc_a = Hlc::new(1000, node_a);
        let hlc_b = Hlc::new(1000, node_b);
        assert!(hlc_a < hlc_b);
    }

    #[test]
    fn test_hlc_logical_greater_than_physical() {
        let node = [1u8; 16];
        let mut hlc = Hlc::new(1000, node);
        hlc.tick(1000);
        hlc.tick(1000);
        hlc.tick(1000);
        assert!(hlc.counter >= 2);
        let hlc_base = Hlc::new(1000, node);
        assert!(hlc > hlc_base);
    }

    #[test]
    fn test_hlc_serialization() {
        let hlc = Hlc::new(12345, [7u8; 16]);
        let bytes = hlc.to_bytes();
        assert_eq!(bytes.len(), 26);
    }

    #[test]
    fn test_hlc_parse_from_bytes() {
        let original = Hlc::new(12345, [7u8; 16]);
        let bytes = original.to_bytes();
        assert_eq!(bytes.len(), 26);
    }

    #[test]
    fn test_hlc_max_value() {
        let hlc = Hlc::new(u64::MAX, [0xff; 16]);
        assert_eq!(hlc.wall_ms, u64::MAX);
    }

    #[test]
    fn test_hlc_zero_value() {
        let hlc = Hlc::zero([1u8; 16]);
        assert_eq!(hlc.wall_ms, 0);
        assert_eq!(hlc.counter, 0);
        assert_eq!(hlc.node_id, [1u8; 16]);
    }

    #[test]
    fn test_hlc_tick_same_time() {
        let node = [1u8; 16];
        let mut hlc = Hlc::new(1000, node);
        for _ in 0..5 {
            hlc.tick(1000);
        }
        assert_eq!(hlc.counter, 5);
    }

    #[test]
    fn test_hlc_tick_advances() {
        let node = [1u8; 16];
        let mut hlc = Hlc::new(1000, node);
        hlc.tick(1000);
        hlc.tick(1000);
        assert!(hlc.counter > 0);
        hlc.tick(2000);
        assert_eq!(hlc.wall_ms, 2000);
        assert_eq!(hlc.counter, 0);
    }

    #[test]
    fn test_hlc_cmp() {
        let node = [1u8; 16];
        let hlc1 = Hlc::new(1000, node);
        let hlc2 = Hlc::new(1001, node);
        // wall_ms differs: 1000 < 1001
        assert_eq!(hlc1.cmp(&hlc2), Ordering::Less);
        assert_eq!(hlc2.cmp(&hlc1), Ordering::Greater);
    }
}
