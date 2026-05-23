use super::*;
use crate::worldgen::tuning::{
    FINE_CACHE_CAP, FINE_CELLS_PER_REGION, FINE_REGION_SIZE, MACRO_REGION_SIZE,
};
use std::sync::Arc;

#[test]
fn region_coord_containing_is_floor_divide() {
    assert_eq!(RegionCoord::containing(0, 0), RegionCoord { x: 0, z: 0 });
    assert_eq!(
        RegionCoord::containing(511, 511),
        RegionCoord { x: 0, z: 0 }
    );
    assert_eq!(
        RegionCoord::containing(512, 512),
        RegionCoord { x: 1, z: 1 }
    );
    assert_eq!(
        RegionCoord::containing(-1, -1),
        RegionCoord { x: -1, z: -1 }
    );
    assert_eq!(
        RegionCoord::containing(-FINE_REGION_SIZE, -FINE_REGION_SIZE),
        RegionCoord { x: -1, z: -1 }
    );
}

#[test]
fn macro_region_coord_containing_is_floor_divide() {
    assert_eq!(
        MacroRegionCoord::containing(0, 0),
        MacroRegionCoord { x: 0, z: 0 }
    );
    assert_eq!(
        MacroRegionCoord::containing(MACRO_REGION_SIZE - 1, 0),
        MacroRegionCoord { x: 0, z: 0 }
    );
    assert_eq!(
        MacroRegionCoord::containing(MACRO_REGION_SIZE, 0),
        MacroRegionCoord { x: 1, z: 0 }
    );
}

#[test]
fn lru_evicts_oldest() {
    let cache = fresh_fine_cache();
    // Insert FINE_CACHE_CAP + 1 entries; the first should be gone.
    for i in 0..=(FINE_CACHE_CAP as i32) {
        let c = RegionCoord { x: i, z: 0 };
        let _ = get_fine(&cache, c, || build_fine_region_placeholder(c));
    }
    let was_evicted = cache.peek(&RegionCoord { x: 0, z: 0 }).is_none();
    assert!(
        was_evicted,
        "expected the first-inserted entry to be evicted after CAP+1 inserts"
    );
    // And the most recent one should still be there.
    assert!(
        cache
            .peek(&RegionCoord {
                x: FINE_CACHE_CAP as i32,
                z: 0
            })
            .is_some()
    );
}

#[test]
fn cache_returns_same_arc_on_repeat_lookup() {
    let cache = fresh_fine_cache();
    let coord = RegionCoord { x: 4, z: 5 };
    let a = get_fine(&cache, coord, || build_fine_region_placeholder(coord));
    let b = get_fine(&cache, coord, || build_fine_region_placeholder(coord));
    assert!(Arc::ptr_eq(&a, &b));

    // peek_fine returns the same Arc without building.
    let c = peek_fine(&cache, coord).expect("peek must find cached");
    assert!(Arc::ptr_eq(&a, &c));

    // peek_fine on cold key returns None and doesn't insert.
    let cold = RegionCoord { x: 999, z: 999 };
    assert!(peek_fine(&cache, cold).is_none());
    assert!(cache.peek(&cold).is_none(), "peek must not insert");
}

#[test]
fn concurrent_lookup_builds_key_once() {
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    let cache = fresh_fine_cache();
    let coord = RegionCoord { x: 7, z: 8 };
    let starts = Arc::new(Barrier::new(8));
    let builds = Arc::new(AtomicUsize::new(0));
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let cache = cache.clone();
            let starts = starts.clone();
            let builds = builds.clone();
            thread::spawn(move || {
                starts.wait();
                get_fine(&cache, coord, || {
                    builds.fetch_add(1, Ordering::SeqCst);
                    thread::sleep(Duration::from_millis(25));
                    build_fine_region_placeholder(coord)
                })
            })
        })
        .collect();

    let first = handles
        .into_iter()
        .map(|h| h.join().expect("worker must not panic"))
        .reduce(|a, b| {
            assert!(Arc::ptr_eq(&a, &b));
            a
        })
        .expect("at least one handle");

    assert!(Arc::ptr_eq(
        &first,
        &peek_fine(&cache, coord).expect("region must be cached")
    ));
    assert_eq!(builds.load(Ordering::SeqCst), 1);
}

#[test]
fn bitset_get_and_set_round_trip() {
    let mut bytes = vec![0u8; 4];
    bitset_set(&mut bytes, 0, true);
    bitset_set(&mut bytes, 7, true);
    bitset_set(&mut bytes, 9, true);
    bitset_set(&mut bytes, 31, true);
    assert!(bitset_get(&bytes, 0));
    assert!(bitset_get(&bytes, 7));
    assert!(!bitset_get(&bytes, 8));
    assert!(bitset_get(&bytes, 9));
    assert!(bitset_get(&bytes, 31));
    bitset_set(&mut bytes, 7, false);
    assert!(!bitset_get(&bytes, 7));
}

#[test]
fn placeholder_region_has_correct_buffer_sizes() {
    let r = FineRegion::empty(RegionCoord { x: 0, z: 0 });
    let n = (FINE_CELLS_PER_REGION * FINE_CELLS_PER_REGION) as usize;
    assert_eq!(r.h_pre.len(), n);
    assert_eq!(r.flow_dir.len(), n);
    assert_eq!(r.flow_acc.len(), n);
    assert_eq!(r.width.len(), n);
}
