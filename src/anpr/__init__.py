"""ANPR prototype - Meridian Access Systems.

An automated number plate reader built from scratch for ISM 6642.

THE ONE RULE OF THIS REPOSITORY
    Logic lives in `src/anpr/`. Notebooks and scripts import it and call it.
    A notebook cell should read `from anpr.models import train_cnn`, never
    contain 200 lines of model code. This is what lets five people work in
    parallel without merge conflicts, and what makes the repo runnable from
    a clean clone (§9 of the brief; failing that check is -8 points).

PACKAGE MAP - one sub-package per team role (§7)
    anpr.config      shared settings, RNG seeding      (everyone)
    anpr.data        EMNIST, plate generation, splits  (Data owner)
    anpr.models      architectures and training        (Model owner)
    anpr.segment     plate image -> character crops    (Pipeline owner)
    anpr.inference   crops -> plate string             (Pipeline owner)
    anpr.evaluate    accuracy, confusion, conditions   (QA owner)
    anpr.business    cost model, trust threshold       (Business owner)

THE PIPELINE, END TO END
    plate image
      -> segment.binarize        threshold to black/white
      -> segment.components      connected components -> N character boxes
      -> data.contract           each box -> canonical 28x28 float32 [0,1]
      -> models (CNN)            each crop -> class + confidence
      -> inference.read_plate    assemble string, aggregate confidence
      -> business.trust_policy   auto-accept, or route to a human?

    Note that the SAME `data.contract` functions run at training time and at
    inference time. That is deliberate: §4 names mismatched preprocessing as
    the single most common cause of "57% validation accuracy, unusable on
    real images". One definition, used in both places, makes the mismatch
    impossible rather than unlikely.
"""

__version__ = "0.1.0"
