# Weekly Progress Log

> Update this file **every week**. Add a new entry at the top for each week.
> This is the first thing we check during review. Keep it honest and specific — it also feeds your attendance record (Rule 1).

**How to use:** copy the *Week template* block below for each new week. Newest week goes at the top.

---

## Week template — copy me

### Week N — YYYY-MM-DD

**Attended this week's meeting:** Yes / No (if No, did you email leave? Yes / No)

**Progress this week**
- _What did you actually do / finish?_

**Challenges & blockers**
- _What got in the way? What are you stuck on?_

**Next steps**
- _What will you do next week?_

**Hours spent (optional):** _e.g. 6h_

**Links (optional):** _commits, notebooks, docs, datasets..._

---

<!-- =================  YOUR ENTRIES BELOW  ================= -->

### Week 2 — 2026-06-15

**Attended this week's meeting:** Yes

**Progress this week**
- implement E and TW constriant to the original POMO approach
- implement E constraint to GA approach
- recreate the OR-Tools method and add ALNS hybrid solution to it
- run the 4 approaches with the same instances and compare the results

**Challenges & blockers**

1. **POMO**:
- Challenge: Model architecture modification (input features 3D→6D, decoder context)
- Solution: Extended encoder, added penalty annealing

2. **GA**:
- Challenge: Feasibility-preserving genetic operators
- Solution: Hard constraint mask (conservative energy estimation)

3. **OR-Tools**:
- Challenge: EV energy constraint not natively supported
- Solution: Approximated with capacity constraints (incomplete)

4. **ALNS**:
- Challenge: Balancing exploration vs. constraint satisfaction
- Solution: Soft constraint tolerance (violation < 5.0)

**Next steps**
- _What will you do next week?_

**Hours spent (optional):** 

**Links (optional):** 


### Week 1 — 2026-06-08

**Attended this week's meeting:** Yes

**Progress this week**
- Set up repository from the FURP template.
- Read OR-Tools routing guides from the website
- Complete the environment configuration
- Pass the smoke test
- Establish a baseline implementation

**Challenges & blockers**
- No major blockers encountered this week

**Next steps**
### Next steps
- Extend the smoke test script to a larger VRP instance to test solver scalability
- Add basic vehicle capacity constraints to the model, following OR-Tools documentation examples
- Implement and validate the baseline model with clear, reproducible output

**Hours spent (optional):**

**Links (optional):**
- OR-Tools website: https://developers.google.com/optimization/routing/vrp?hl=zh-cn
