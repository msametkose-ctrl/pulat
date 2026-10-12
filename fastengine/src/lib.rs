type S = [i8; 28];

fn single(b: &S, d: i8, out: &mut Vec<S>) {
    if b[24] > 0 {
        let t = (24 - d) as usize;
        if b[t] >= -1 {
            let mut n = *b;
            n[24] -= 1;
            if n[t] == -1 {
                n[t] = 0;
                n[25] += 1;
            }
            n[t] += 1;
            out.push(n);
        }
        return;
    }
    let mut top: i32 = -1;
    for i in (0..24).rev() {
        if b[i] > 0 {
            top = i as i32;
            break;
        }
    }
    let can_bear = top < 6;
    let mut i = top;
    while i >= 0 {
        let iu = i as usize;
        if b[iu] > 0 {
            let t = i - d as i32;
            if t >= 0 {
                let tu = t as usize;
                if b[tu] >= -1 {
                    let mut n = *b;
                    n[iu] -= 1;
                    if n[tu] == -1 {
                        n[tu] = 0;
                        n[25] += 1;
                    }
                    n[tu] += 1;
                    out.push(n);
                }
            } else if can_bear && (t == -1 || i == top) {
                let mut n = *b;
                n[iu] -= 1;
                n[26] += 1;
                out.push(n);
            }
        }
        i -= 1;
    }
}

fn legal(s: &S, d1: i8, d2: i8, out: &mut Vec<S>) {
    let (hi, lo) = (d1.max(d2), d1.min(d2));
    let orders: &[&[i8]] = if d1 == d2 { &[&[d1, d1, d1, d1]] } else { &[&[hi, lo], &[lo, hi]] };
    let mut best = 0;
    let mut res: Vec<S> = Vec::new();
    let mut nxt: Vec<S> = Vec::with_capacity(256);
    for order in orders {
        let mut level = vec![*s];
        let mut depth = 0;
        for &d in order.iter() {
            nxt.clear();
            for st in &level {
                single(st, d, &mut nxt);
            }
            if nxt.is_empty() {
                break;
            }
            nxt.sort_unstable();
            nxt.dedup();
            std::mem::swap(&mut level, &mut nxt);
            depth += 1;
        }
        if depth > best {
            best = depth;
            res = level;
        } else if depth == best && depth > 0 && !(depth == 1 && d1 != d2) {
            res.extend(level);
        }
    }
    if best == 0 {
        out.push(*s);
        return;
    }
    res.sort_unstable();
    res.dedup();
    out.extend(res);
}

#[no_mangle]
pub unsafe extern "C" fn gen_batch(
    n: usize,
    states: *const i8,
    dice: *const u8,
    out: *mut i8,
    cap: usize,
    counts: *mut u32,
    nthreads: usize,
) -> i64 {
    let states = std::slice::from_raw_parts(states as *const S, n);
    let dice = std::slice::from_raw_parts(dice, 2 * n);
    let counts = std::slice::from_raw_parts_mut(counts, n);
    let nt = nthreads.max(1).min(n.max(1));
    let chunk = (n + nt - 1) / nt.max(1);
    let parts: Vec<(Vec<S>, Vec<u32>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..nt)
            .map(|k| {
                let lo = (k * chunk).min(n);
                let hi = ((k + 1) * chunk).min(n);
                sc.spawn(move || {
                    let mut v = Vec::with_capacity((hi - lo) * 32);
                    let mut c = Vec::with_capacity(hi - lo);
                    for g in lo..hi {
                        let before = v.len();
                        legal(&states[g], dice[2 * g] as i8, dice[2 * g + 1] as i8, &mut v);
                        c.push((v.len() - before) as u32);
                    }
                    (v, c)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let total: usize = parts.iter().map(|p| p.0.len()).sum();
    if total > cap {
        return -(total as i64);
    }
    let out = std::slice::from_raw_parts_mut(out as *mut S, cap);
    let (mut o, mut g) = (0, 0);
    for (v, c) in parts {
        out[o..o + v.len()].copy_from_slice(&v);
        o += v.len();
        counts[g..g + c.len()].copy_from_slice(&c);
        g += c.len();
    }
    total as i64
}
