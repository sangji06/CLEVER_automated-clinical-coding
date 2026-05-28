# CLEVER Annotation App

This Streamlit app was used to support ground-truth annotation for the CLEVER study.
It is included as a research utility, separate from the main CLEVER inference pipeline.

## Local Use

Place local clinical text files in `data/` and run from this directory:

```bash
cd tools/annotation_app
streamlit run app.py
```

The app writes intermediate annotation outputs to `output/` and tracks completed files in `progress.json`.
Do not commit clinical notes, annotation outputs, or progress files.

## Notes

The public repository does not include patient records or ground-truth annotations.
The app is provided for transparency and local reuse with authorized data only.

