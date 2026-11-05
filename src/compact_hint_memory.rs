//! Compact-hint memory: a dependency-free Hebbian memory for pointer-style
//! retrieval.
//!
//! A *pointer* is a small record whose whole job is to be found at the moment
//! it is needed, so the consumer can then load the full thing (a skill body, a
//! tool schema, a doc). That is the opposite of an always-on catalog: nothing
//! is rendered into the prompt up front, and only the handful of pointers that
//! survive the gate are ever named.
//!
//! Storage is a plain map of pointers plus a map of Hebbian edges. There is no
//! vector index on purpose: recall is a lexical prior blended with usage
//! strength, which keeps the gate's decision in the caller's hands (a
//! server-side similarity score is normally relative-normalised and is not a
//! usable confidence signal).
//!
//! Semantics:
//! - store: insert or update a pointer by name (caller-supplied `id`).
//! - recall: tokenise the query, count the mass scores, require half the query
//!   terms to land (or all of them for a single-term query), then rank by
//!   `lexical * (0.5 + 0.5 * hebb) * (0.6 + 0.4 * decay)`. Every recalled
//!   pointer is reinforced.
//! - reinforce: reorder the two or three pointers worth loading next as a trail
//!   from previous to next (`HebbianEdge`), reinforcing edges that were
//!   traversed together.
//! - decay: exponential relaxation towards zero on a half-life, and forget on
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Ordered, weighted trail between two pointers.
#[derive(Debug, Clone, PartialEq)]
pub struct HebbianEdge {
    pub previous: String,
    pub next: String,
    pub weight: f64,
}

/// A pointer kept in memory.
#[derive(Debug, Clone)]
pub struct Pointer {
    pub id: String,
    pub text: String,
    pub tags: Vec<String>,
    pub importance: f64,
    pub strength: f64,
    pub access_count: u32,
    pub last_access: f64,
    pub created_at: f64,
}

/// A recall result: the pointer plus its blended score.
#[derive(Debug, Clone)]
pub struct Hit {
    pub id: String,
    pub text: String,
    pub score: f64,
    pub lexical: f64,
}

const DECAY_HALF_LIFE_S: f64 = 30.0 * 24.0 * 3600.0;
const EDGE_STEP: f64 = 0.08;
const EDGE_CEIL: f64 = 1.0;
const STRENGTH_STEP: f64 = 0.05;
const STRENGTH_CEIL: f64 = 5.0;

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect::<Vec<_>>()
        .into_iter()
        .map(|t| {
            if t.len() >= 5 && t.ends_with('s') && !t.ends_with("ss") {
                t[..t.len() - 1].to_string()
            } else {
                t
            }
        })
        .collect()
}

#[derive(Default)]
pub struct CompactHintMemory {
    pointers: HashMap<String, Pointer>,
    edges: HashMap<(String, String), f64>,
}

impl CompactHintMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.pointers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pointers.is_empty()
    }

    /// Insert or replace a pointer by id.
    pub fn store(&mut self, id: &str, text: &str, tags: &[String], importance: f64) {
        let now = now_secs();
        match self.pointers.get_mut(id) {
            Some(p) => {
                p.text = text.to_string();
                p.tags = tags.to_vec();
                p.importance = importance;
            }
            None => {
                self.pointers.insert(
                    id.to_string(),
                    Pointer {
                        id: id.to_string(),
                        text: text.to_string(),
                        tags: tags.to_vec(),
                        importance,
                        strength: 1.0,
                        access_count: 0,
                        last_access: 0.0,
                        created_at: now,
                    },
                );
            }
        }
    }

    pub fn forget(&mut self, id: &str) -> bool {
        let removed = self.pointers.remove(id).is_some();
        self.edges
            .retain(|(a, b), _| a != id && b != id);
        removed
    }

    /// Recall pointers relevant to `query`, strongest first, capped at `limit`.
    /// Hits are reinforced (strength, access count) and sharing-tag groups get
    /// their edges strengthened, exactly as a co-activation would.
    pub fn recall(&mut self, query: &str, limit: usize) -> Vec<Hit> {
        let q = tokenize(query);
        if q.is_empty() || self.pointers.is_empty() || limit == 0 {
            return Vec::new();
        }
        let qset: Vec<String> = {
            let mut v = q.clone();
            v.sort();
            v.dedup();
            v
        };
        let now = now_secs();

        let mut scored: Vec<(f64, f64, Hit, Vec<String>)> = Vec::new();
        for p in self.pointers.values() {
            let doc = tokenize(&p.text);
            if doc.is_empty() {
                continue;
            }
            let hits: Vec<&String> = qset.iter().filter(|t| doc.contains(t)).collect();
            if hits.is_empty() {
                continue;
            }
            let cov = hits.len() as f64 / qset.len() as f64;
            let dens = hits.len() as f64 / (doc.len() as f64).sqrt().max(1.0);
            let lexical = 0.65 * cov + 0.35 * dens;
            let age = (now - p.last_access.max(p.created_at)).max(0.0);
            let decay = 0.5_f64.powf(age / DECAY_HALF_LIFE_S);
            let hebb = (1.0 + (p.access_count as f64).ln_1p().max(0.0)) * p.strength;
            let score = lexical * (0.5 + 0.5 * hebb) * (0.6 + 0.4 * decay) + 0.05 * p.importance;
            scored.push((
                score,
                cov,
                Hit { id: p.id.clone(), text: p.text.clone(), score, lexical },
                p.tags.clone(),
            ));
        }

        let need = if qset.len() <= 1 { 1.0 } else { 0.5 };
        let any_qualifies = scored.iter().any(|s| s.1 >= need);
        if any_qualifies {
            scored.retain(|s| s.1 >= need);
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        let ids: Vec<String> = scored.iter().map(|s| s.2.id.clone()).collect();
        self.reinforce(&ids);

        scored
            .into_iter()
            .map(|mut s| {
                s.2.score = (s.2.score * 10_000.0).round() / 10_000.0;
                s.2
            })
            .collect()
    }

    /// Strengthen the traversal trail between consecutive recalled pointers and
    /// the shared-tag edges between all recalled pointers.
    pub fn reinforce(&mut self, ids: &[String]) {
        let now = now_secs();
        let mut tags_of: Vec<(String, Vec<String>)> = Vec::new();
        for id in ids {
            if let Some(p) = self.pointers.get_mut(id) {
                p.strength = (p.strength + STRENGTH_STEP).min(STRENGTH_CEIL);
                p.access_count = p.access_count.saturating_add(1);
                p.last_access = now;
                tags_of.push((id.clone(), p.tags.clone()));
            }
        }
        for w in ids.windows(2) {
            self.bump_edge(&w[0], &w[1], EDGE_STEP);
        }
        for (i, (a, atags)) in tags_of.iter().enumerate() {
            for (b, btags) in tags_of.iter().skip(i + 1) {
                if a != b && atags.iter().any(|t| btags.contains(t)) {
                    self.bump_edge(a, b, EDGE_STEP);
                }
            }
        }
    }

    fn bump_edge(&mut self, a: &str, b: &str, step: f64) {
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        let key = (a.to_string(), b.to_string());
        let entry = self.edges.entry(key).or_insert(0.0);
        *entry = (*entry + step).min(EDGE_CEIL);
    }

    /// Read the current trail weight between two pointers.
    pub fn edge_weight(&self, a: &str, b: &str) -> f64 {
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        self.edges
            .get(&(a.to_string(), b.to_string()))
            .copied()
            .unwrap_or(0.0)
    }

    /// All edges, strongest first.
    pub fn trails(&self) -> Vec<HebbianEdge> {
        let mut v: Vec<HebbianEdge> = self
            .edges
            .iter()
            .map(|((a, b), w)| HebbianEdge {
                previous: a.clone(),
                next: b.clone(),
                weight: *w,
            })
            .collect();
        v.sort_by(|x, y| y.weight.partial_cmp(&x.weight).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    /// Exponential relaxation of every pointer's strength on the half-life.
    /// Returns the number of pointers that dropped below `forget_floor` and were
    /// removed.
    pub fn decay(&mut self, elapsed: Duration, forget_floor: f64) -> usize {
        let factor = 0.5_f64.powf(elapsed.as_secs_f64() / DECAY_HALF_LIFE_S);
        let mut dead = Vec::new();
        for p in self.pointers.values_mut() {
            p.strength *= factor;
            if p.strength < forget_floor {
                dead.push(p.id.clone());
            }
        }
        for id in &dead {
            self.forget(id);
        }
        dead.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem() -> CompactHintMemory {
        let mut m = CompactHintMemory::new();
        m.store(
            "browser-ops",
            "browser-ops :: drive a real browser over CDP when sites block web tools :: web",
            &["web".into()],
            0.5,
        );
        m.store(
            "invoice-pdf",
            "invoice-pdf :: render invoice workbook rows into a signed PDF letterhead :: finance",
            &["finance".into()],
            0.5,
        );
        m.store(
            "patio-rig",
            "patio-rig :: start or fix the patio ghost voice rig :: voice",
            &["voice".into()],
            0.5,
        );
        m
    }

    #[test]
    fn rank_order_and_reinforcement() {
        let mut m = mem();
        let hits = m.recall("browser blocks the site, drive chrome", 3);
        assert!(hits.iter().any(|h| h.id == "browser-ops"));
        let h1 = m.recall("start or fix the patio voice rig", 1);
        assert_eq!(h1[0].id, "patio-rig");
        let _ = h1;
    }

    #[test]
    fn noise_recall_is_restrained() {
        let mut m = mem();
        let h = m.recall("invoice", 3);
        assert_eq!(h[0].id, "invoice-pdf");
    }

    #[test]
    fn co_recall_forms_a_trail() {
        let mut m = mem();
        let ids = vec!["patio-rig".to_string(), "browser-ops".to_string()];
        m.reinforce(&ids);
        assert!(m.edge_weight("patio-rig", "browser-ops") > 0.0);
        assert_eq!(m.trails().len(), 1);
    }

    #[test]
    fn decay_forgets_weak_pointers() {
        let mut m = mem();
        let dead = m.decay(Duration::from_secs(30 * 24 * 60 * 60 * 200), 0.6);
        assert!(dead > 0);
    }

    #[test]
    fn forget_drops_pointer() {
        let mut m = mem();
        assert!(m.forget("patio-rig"));
        assert!(m.recall("patio rig", 3).iter().all(|h| h.id != "patio-rig"));
        assert_eq!(m.len(), 2);
    }
}
