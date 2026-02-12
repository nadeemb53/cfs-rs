//! Hybrid Logical Clock (HLC) for causal ordering
//!
//! Per CFS-003 §7: HLC combines wall-clock time with a logical counter
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Per CFS-003 §7:
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
    /// Per CFS-003 §7:
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
            self.counter = remote.counter.saturating_add(1);
        } else {
            // Wall clock is ahead of both: reset counter
            self.counter = 0;
        }

        self.wall_ms = max_wall;
    }

    /// Serialize to bytes for deterministic hashing/signing.
    ///
    /// Layout: wall_ms (8 LE) || counter (2 LE) || node_id (16) = 26 bytes
    pub fn to_bytes(&self) -> [u8; 26] {
        let mut bytes = [0u8; 26];
        bytes[0..8].copy_from_slice(&self.wall_ms.to_le_bytes());
        bytes[8..10].copy_from_slice(&self.counter.to_le_bytes());
        bytes[10..26].copy_from_slice(&self.node_id);
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8; 26]) -> Self {
        let wall_ms = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let counter = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
        let mut node_id = [0u8; 16];
        node_id.copy_from_slice(&bytes[10..26]);
        Self {
            wall_ms,
            counter,
            node_id,
        }
    }
}

impl PartialEq for Hlc {
    fn eq(&self, other: &Self) -> bool {
        self.wall_ms == other.wall_ms
            && self.counter == other.counter
            && self.node_id == other.node_id
    }
}

impl Eq for Hlc {}

impl PartialOrd for Hlc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hlc {
    fn cmp(&self, other: &Self) -> Ordering {
        self.wall_ms
            .cmp(&other.wall_ms)
            .then_with(|| self.counter.cmp(&other.counter))
            .then_with(|| self.node_id.cmp(&other.node_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_node() -> [u8; 16] {
        [1u8; 16]
    }

    fn other_node() -> [u8; 16] {
        [2u8; 16]
    }

    #[test]
    fn test_hlc_tick_advances_wall() {
        let mut hlc = Hlc::new(1000, test_node());
        hlc.tick(2000);
        assert_eq!(hlc.wall_ms, 2000);
        assert_eq!(hlc.counter, 0);
    }

    #[test]
    fn test_hlc_tick_increments_counter_on_same_wall() {
        let mut hlc = Hlc::new(1000, test_node());
        hlc.tick(1000); // Same wall time
        assert_eq!(hlc.wall_ms, 1000);
        assert_eq!(hlc.counter, 1);

        hlc.tick(999); // Wall time went backwards
        assert_eq!(hlc.wall_ms, 1000);
        assert_eq!(hlc.counter, 2);
    }

    #[test]
    fn test_hlc_merge_remote_ahead() {
        let mut local = Hlc::new(1000, test_node());
        let remote = Hlc {
            wall_ms: 2000,
            counter: 5,
            node_id: other_node(),
        };

        local.merge(&remote, 1500);
        assert_eq!(local.wall_ms, 2000);
        assert_eq!(local.counter, 6); // remote.counter + 1
    }

    #[test]
    fn test_hlc_merge_local_ahead() {
        let mut local = Hlc::new(3000, test_node());
        local.counter = 3;
        let remote = Hlc::new(2000, other_node());

        local.merge(&remote, 2500);
        assert_eq!(local.wall_ms, 3000);
        assert_eq!(local.counter, 4); // local.counter + 1
    }

    #[test]
    fn test_hlc_merge_now_ahead() {
        let mut local = Hlc::new(1000, test_node());
        let remote = Hlc::new(2000, other_node());

        local.merge(&remote, 5000); // now is ahead of both
        assert_eq!(local.wall_ms, 5000);
        assert_eq!(local.counter, 0); // reset
    }

    #[test]
    fn test_hlc_merge_all_equal() {
        let mut local = Hlc {
            wall_ms: 1000,
            counter: 3,
            node_id: test_node(),
        };
        let remote = Hlc {
            wall_ms: 1000,
            counter: 5,
            node_id: other_node(),
        };

        local.merge(&remote, 1000);
        assert_eq!(local.wall_ms, 1000);
        assert_eq!(local.counter, 6); // max(3, 5) + 1
    }

    #[test]
    fn test_hlc_total_ordering() {
        let a = Hlc {
            wall_ms: 1000,
            counter: 0,
            node_id: test_node(),
        };
        let b = Hlc {
            wall_ms: 1000,
            counter: 1,
            node_id: test_node(),
        };
        let c = Hlc {
            wall_ms: 2000,
            counter: 0,
            node_id: test_node(),
        };

        assert!(a < b);
        assert!(b < c);
        assert!(a < c);
    }

    #[test]
    fn test_hlc_node_id_tiebreaker() {
        let a = Hlc {
            wall_ms: 1000,
            counter: 0,
            node_id: [1u8; 16],
        };
        let b = Hlc {
            wall_ms: 1000,
            counter: 0,
            node_id: [2u8; 16],
        };

        assert!(a < b); // node_id [1;16] < [2;16]
    }

    #[test]
    fn test_hlc_bytes_roundtrip() {
        let hlc = Hlc {
            wall_ms: 1234567890123,
            counter: 42,
            node_id: [7u8; 16],
        };

        let bytes = hlc.to_bytes();
        assert_eq!(bytes.len(), 26);

        let recovered = Hlc::from_bytes(&bytes);
        assert_eq!(hlc, recovered);
    }

    #[test]
    fn test_hlc_zero() {
        let hlc = Hlc::zero(test_node());
        assert_eq!(hlc.wall_ms, 0);
        assert_eq!(hlc.counter, 0);
        assert_eq!(hlc.node_id, test_node());
    }
}
