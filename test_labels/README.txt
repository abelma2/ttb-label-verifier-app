Put label images here to test the pipeline.

- Drop the FRONT and BACK/"other" labels of the same product in here
  (e.g. captain_johns_front.png and captain_johns_back.png). The government
  warning, net contents, and name/address are usually on the back, so include it.
- Supported types: .png, .jpg, .jpeg

This folder is gitignored — images you drop here are never committed.

To validate (after putting your OpenAI key in .streamlit/secrets.toml):

  # treat ALL images in this folder as ONE product (front + back):
  python scripts/smoke_test.py test_labels

  # or test specific images as one product:
  python scripts/smoke_test.py test_labels/front.png test_labels/back.png

  # or test each image as its own separate product:
  python scripts/smoke_test.py --each test_labels

Or run the full app and upload through the browser:

  streamlit run app.py
