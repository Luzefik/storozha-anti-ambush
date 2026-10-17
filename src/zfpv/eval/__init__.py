"""Model-agnostic evaluation: every model is wrapped as a Detector and scored by ONE
shared evaluator, never by a library's own built-in mAP printout."""
