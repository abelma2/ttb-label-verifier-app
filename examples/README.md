# Example label — try the app in one minute

A synthetic, compliant malt-beverage label (fictional brand) you can use to see the
verifier work without hunting for your own bottle photos.

| File | What it is |
| --- | --- |
| `malt_and_hop_Front.jpg` | front label (brand, class, ABV, net contents, address) |
| `malt_and_hop_Other.jpg` | back label (government warning lives here) |
| `malt_and_hop_application.json` | the values "the applicant submitted", for label-vs-application matching |

**Quick demo (single mode):** upload `malt_and_hop_Front.jpg` as the front and
`malt_and_hop_Other.jpg` as the back, leave the form blank, and verify — that's the
rules-only screening. Every field should come back green (the bold read on the warning
occasionally lands on "needs review"; that's the fail-closed design, not a bug).

**Label-vs-application matching:** same uploads, but load
`malt_and_hop_application.json` into the application form first (it auto-matches by the
filename stem `malt_and_hop`), then verify. Now each field is also compared against the
submitted value. Edit a value (e.g. change the ABV to 7%) and re-run to see a mismatch
get caught.

**Batch mode:** drop both images in together — they pair into one product by filename —
and optionally add the JSON as the application file.
