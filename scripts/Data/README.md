# Data Overview

## MNE Data

### EEGBCI ()

```python
download_EEGBCI()
```

This is a motor imagery dataset and it has a couple of important features:

**PARTICIPANTS**:

- This dataset contains data from 109 participants. Can be in the range of 1-109 (inclusive).

**EVENTS**:

- T0 corresponds to rest
- T1 corresponds to onset of motion (real or imagined) of
  - the left fist (in runs 3, 4, 7, 8, 11, and 12)
  - both fists (in runs 5, 6, 9, 10, 13, and 14)
- T2 corresponds to onset of motion (real or imagined) of
  - the right fist (in runs 3, 4, 7, 8, 11, and 12)
  - both feet (in runs 5, 6, 9, 10, 13, and 14)

**RUNS**:

1. Baseline, eyes open
2. Baseline, eyes closed
3. Task 1 (open and close left or right fist)
4. Task 2 (imagine opening and closing left or right fist)
5. Task 3 (open and close both fists or both feet)
6. Task 4 (imagine opening and closing both fists or both feet)
7. Task 1
8. Task 2
9. Task 3
10. Task 4
11. Task 1
12. Task 2
13. Task 3
14. Task 4

| run | task |
| --- | --- |
| 1 | baseline, eyes open |
| 2 | baseline, eyes closed |
| 3, 7, 11 | Motor execution: left vs right hand |
| 4, 8, 12 | Motor imagery: left vs right hand |
| 5, 9, 13 | Motor execution: hands vs feet |
| 6, 10, 14 | Motor imagery: hands vs feet |

------
