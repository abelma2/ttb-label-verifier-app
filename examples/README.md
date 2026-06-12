# Example labels — try the app in a minute

Ready-to-use label images covering **spirits, wine, and beer** (plus an import), so you can
see the verifier work without hunting for your own bottle photos. Some are single flat
images; some are front + back pairs (read together as one label). A ready-made application
spreadsheet (`example-labels-applications.xlsx`) lets you exercise label-vs-application
matching without typing anything.

| File(s) | Beverage | What it is / what it exercises |
| --- | --- | --- |
| `abc_distillery_whiskey.png` | spirits | whiskey, single flat image — straightforward spirits screening (ABV mandatory) |
| `abc_distillery_spiced_rum.png` | spirits | spiced rum, single flat image — class, ABV **and proof** (30% / 60 proof), net contents, address, government warning |
| `12345_imports_coconut_rum_liqueur.jpg` | spirits (import) | imported coconut-rum liqueur, single flat image — exercises the country-of-origin / "Produced in Canada" import case |
| `captain_johns_spiced_rum_Front.png` + `captain_johns_spiced_rum_Other.png` | spirits | spiced rum, front + back pair — ABV **and proof** (20% / 40 proof) |
| `lighthouse_stormchaser_white_chardonnay_Front.png` + `..._Other.png` | wine | Chardonnay, front + back pair — the varietal triggers the wine-only **appellation** check |
| `malt_hop_honey_huckleberry_pie_Front.png` + `..._Other.png` | beer | flavored ale, front + back pair — net contents `1 PINT 0.9 FL. OZ.` exercises the volume parser; ABV optional for beer |
| `example-labels-applications.xlsx` | — | filled-in application spreadsheet: **one row per product above**, for the label-vs-application matching demo |

**Quick demo (single mode):** upload `abc_distillery_whiskey.png` as the front label, leave
the form blank, and verify — that's the rules-only screening. For a paired set, upload the
`_Front` file as the front and the `_Other` file as the back; both images are read together
as one label.

**Label-vs-application matching:**

- *Single mode* — type the applicant's values into the form before verifying. For
  `abc_distillery_spiced_rum.png`, try brand `ABC Distillery`, class `Spiced Rum`, alcohol
  `30% Alc/Vol (60 Proof)`, net contents `750 mL`. Then change one value and re-run to watch
  the mismatch get caught.
- *Batch mode* — this is what `example-labels-applications.xlsx` is for: drop all the images
  in together and attach the spreadsheet, and each product is matched against its row. The
  file is just a filled-in copy of the in-app template (one row per product; only the
  `product` column is required, and it must match the image filename stem).

**Batch mode:** drop several files in together. Files pair into products by filename stem
(`captain_johns_spiced_rum_Front` + `captain_johns_spiced_rum_Other` become one product
`captain_johns_spiced_rum`), and each single flat image is its own product. Attach
`example-labels-applications.xlsx` to also screen against the submitted values, or omit it
for rules-only screening.
