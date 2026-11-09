# Problem Definition

## Research Question

Infer a **task graph** for a procedural activity from repeated demonstrations of that activity. A task graph is a directed acyclic graph whose nodes are the procedure's key-steps (plus placeholder START and END nodes) and whose edges are prerequisite relations: an edge says that one key-step must be completed before another can be performed.

The supervision is weak: for each procedure the learner sees only demonstrations, i.e. sequences of key-steps in the order in which people performed them. It must recover the prerequisite structure without being trained on graph labels.

## Why It Matters

Task graphs are an explicit, human-readable representation of the partial order of a procedure. They support downstream video-understanding tasks such as detecting procedural mistakes, predicting missing or upcoming steps, and checking whether a procedure was executed correctly.

The difficulty is that procedures allow many valid orders and that a handful of demonstrations shows only some of them. Steps can co-occur in a fixed order in every demonstration by coincidence, without one being a prerequisite of the other. Real demonstrations are also noisy: a step can be missed, inserted where it does not belong, or recognised as the wrong step.

## Settings

1. **CaptainCook4D**: 24 cooking procedures (7-25 key-steps each) with 1-13 demonstrations each and a ground-truth graph per procedure. 12 procedures have public graphs; the other 12 are held out.
2. **Noisy demonstrations**: procedures whose graphs have the size and shape of CaptainCook4D graphs. Each has 4-10 demonstrations in which every key-step was, with probability 0.05, replaced by another step, deleted, or preceded by an inserted random step (repeated steps are then dropped, keeping the first occurrence). 100 procedures have public graphs; 100 are held out.

## Success Criteria

A successful solution should:
1. recover accurate graphs from demonstrations, measured against held-out ground-truth prerequisite edges;
2. handle procedures with many valid execution orders and only a few demonstrations;
3. stay accurate when demonstrations are noisy;
4. work on every procedure, not only on the easy ones;
5. preserve the weak-supervision setting: demonstrations are the learning signal, graph annotations are for evaluation.
