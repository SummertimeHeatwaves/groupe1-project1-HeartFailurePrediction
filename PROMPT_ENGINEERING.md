# Prompt Engineering Documentation

## Task Selected: Memory Optimization Function (optimize_memory)

### Context
We used AI-assisted development (ChatGPT / GitHub Copilot) to help implement the `optimize_memory(df)` function in `data_processing.py`.

### Prompt 1: Initial Attempt (Vague)
**Prompt:** "Write a function that optimizes memory of a dataframe"
**Result:** A 5-line function that only converted float64 to float32. Missing: int optimization, logging, documentation, error handling.
**Assessment:** Too vague. The AI produced minimal output because the requirements were not specified.

### Prompt 2: Refined (Specific requirements)
**Prompt:** "Write a Python function called optimize_memory that takes a pandas DataFrame as input and returns an optimized DataFrame. Requirements:
1. Convert float64 columns to float32
2. Convert int64 to the smallest int type (int8/int16/int32) based on the actual min/max values in each column
3. Log memory usage before and after using Python logging module
4. Calculate and log the percentage reduction
5. Include type hints, docstring with Args/Returns, and an example
6. Handle edge cases (empty DataFrame, non-numeric columns)"

**Result:** Complete function matching all 6 requirements. Included proper type annotations and comprehensive docstring.
**Assessment:** Specifying numbered requirements produced a much better result. Each requirement mapped to specific code.

### Prompt 3: Test Generation
**Prompt:** "Write pytest unit tests for this optimize_memory function. Include:
- Test that memory is reduced
- Test that float values are preserved within float32 precision
- Test that integer values are exactly preserved
- Test return type is DataFrame
- Test that shape is unchanged
- Test that float64 columns become float32"

**Result:** A well-structured TestOptimizeMemory class with 6 methods.
**Assessment:** Specifying individual test cases ensures comprehensive coverage instead of generic "write tests for this function."

### Lessons Learned
| Lesson | Example |
|---|---|
| Be specific | "Numbered requirements" > "make it good" |
| Include constraints | "Use logging, type hints, docstrings" |
| Ask for tests separately | Better coverage and structure |
| Iterate and refine | Each prompt builds on the previous |
| Provide context | "For a medical ML project" helps tone/rigor |

### Conclusion
Prompt engineering is most effective when you treat the AI as a collaborator who needs clear specifications. Vague prompts produce vague results. Structured, specific prompts with explicit requirements produce production-quality code.