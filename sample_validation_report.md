# MarkItDown validation report

- PDF: `sample.pdf`
- Markdown: `damaged.md`
- **Text coverage before fix: 90.29%**
- Missing blocks found: 4 (0 low-signal: math/axis labels)
- Appended to output: yes
- `(cid:NN)` glyph artifacts in MD: 41
- MD lines with no PDF support (possible noise): 2

## Coverage by page

| Page | Coverage % | Images (not validatable) |
|---|---|---|
| 1 | 92.67 | 0 |
| 2 | 86.58 | 0 |
| 3 | 83.64 | 8 |
| 4 | 87.62 | 35 |
| 5 | 100.0 | 0 |

## Missing content

- p.1 [missing]: which is equal to soft thresholding in the case of f (x) = λ(cid:107)Ψx(cid:107)1 and orthogonal Ψ. For certain sensing matrices and given mild conditions on f (x), AMP’s state evolution guarantees th
- p.2 [missing]: For AMP, VAMP and OAMP, (4) states that the effective noise rk − x0 is white, so can be fully characterised by a scalar τk. This is appropriate for the kind of uniform measurement matrices and separab
- p.3 [missing]: crucial weighting by P −1. As an intuitive example, consider k = 0, where r0 = ΨΦH P −1y. Since EΩ,ε(P −1y) = y0, it follows that r0 is unbiased: E(r0) = w0. Such a rescaling is referred to as “densit
- p.4 [missing]: The efﬁcacy of automatic threshold selection with SURE depends on how accurately the diagonal of Σ2 k from (6) is modelled by τk. For k = 0, 1, . . . , 20, Fig. 5 shows the ground truth NMSE of the wa

## Possible noise in Markdown (informational)

- 012345678910Time (s)-45-40-35-30-25-20-15-10-5NMSE (dB)VDAMP, 4xFISTA, 4xSURE-IT, 4xVDAMP, 6xFISTA, 6xSURE-IT, 6xVDAMP, 8xFISTA, 8xSURE-IT, 8x(a)x0(b)FISTAˆx(c)
- (a) r0−w0 (b) r1−w0 (c) r2−w0 -505Standard normal quantiles-1-0.500.51Quantiles of input sampleDiagonal, scale 1, k=0-505Standard normal quantiles-1-0.500.51Qua
