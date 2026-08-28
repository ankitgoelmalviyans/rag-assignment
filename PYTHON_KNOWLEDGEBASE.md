# Python Knowledgebase — RAG Assignment

A running reference of every Python term/function/concept used in this project so far,
explained simply, with the closest C#/.NET comparison and a concrete example. Updated as new
concepts appear.

---

## Installing Python — Windows vs WSL2, and why this project uses WSL2

**Windows (a typical .NET dev machine)** — two common routes:
- **python.org installer (recommended)** — download from python.org, run it, and make sure to
  tick **"Add python.exe to PATH"** during setup. After this, `python --version` works directly
  in PowerShell/cmd.
- **winget** — `winget install Python.Python.3.12`
- **Avoid the Microsoft Store version for dev work** — Windows ships a fake `python.exe`
  "app execution alias" that, if nothing else is installed, prints *"Python was not found; run
  without arguments to install from the Microsoft Store"* — this is exactly the error hit
  earlier in this project when a command tried to run `python` from a plain Windows shell with
  no real interpreter installed there. It's a placeholder, not a real Python install.

**WSL2 (Ubuntu)** — what this project actually runs on:
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```
Ubuntu images typically ship `python3` already; `python3-venv` (for creating virtual
environments) and `python3-pip` sometimes need installing separately.

**Why this project specifically uses WSL2** rather than Windows-native Python: the project's
`.venv` was created inside WSL (its folder layout — `.venv/lib/python3.12/site-packages/...` —
is the Linux venv shape, not Windows' `.venv\Scripts\...`), so Ollama, FAISS, and every script
here run through a real Linux environment rather than Windows — smoother native package
compatibility, and matches how Ollama itself is typically run for local dev.

**Why `python` works on Windows but only `python3` exists by default on Ubuntu** — this isn't
a project-specific quirk, it's standard behavior on Debian/Ubuntu-family Linux distros: they
deliberately ship only a `python3` command system-wide, to avoid ambiguity with the old Python 2
`python` command from years ago. A package called `python-is-python3` exists to add the `python`
alias system-wide, but isn't installed by default. Windows' installer, by contrast, has always
just created a plain `python` command.

**Why `python` *does* work once inside an activated `.venv`, on either OS** — when you create a
venv with `python3 -m venv .venv`, Python's own `venv` module always creates **both** `python`
and `python3` inside `.venv/bin/` (or `.venv\Scripts\` on Windows), regardless of what the base
system provides. Then:
```bash
source .venv/bin/activate      # WSL/Linux/macOS
.venv\Scripts\activate         # Windows
```
doesn't install anything new — it just **temporarily prepends the venv's own bin/Scripts folder
to your shell's `PATH`** environment variable for that session. Since your shell searches `PATH`
folders in order, `python` now resolves to the venv's copy before it ever reaches the system
folders. Run `deactivate`, and `PATH` reverts, so `python` goes back to whatever (or nothing) the
base OS provides on its own.
*C# equivalent:* similar in spirit to how `dotnet` resolves which SDK to use based on `PATH`/
`global.json` — the venv is just temporarily reordering the search path, not installing anything
permanent.

---

## Project & environment concepts

**`pip`** — Python's package installer. Downloads and installs libraries.
*C# equivalent:* NuGet.
```bash
pip install pypdf requests faiss-cpu numpy
```
```
# C#: Install-Package or dotnet add package
dotnet add package Newtonsoft.Json
```

**`.venv` (virtual environment)** — an isolated, project-local set of installed packages, so
this project's dependencies don't clash with any other Python project on your machine.
*C# equivalent:* no exact match, but similar in spirit to keeping a project's package
references scoped to that project rather than installed machine-wide.
```bash
python3 -m venv .venv        # create it
source .venv/bin/activate    # turn it on — prompt changes to (.venv) ...
```

**Package / module / class / object** — the mental model used throughout:
```
package/module (a .py file or folder of .py files, e.g. "pypdf")
        |
        +-- class (e.g. PdfReader)
              |
              +-- object/instance (e.g. reader = PdfReader(pdf_file))
```
```python
from pypdf import PdfReader      # package -> class
reader = PdfReader(pdf_file)     # class -> object/instance
```
```csharp
using PdfLibrary;                          // similar role to "from ... import"
var reader = new PdfReader(pdfFile);       // class -> object/instance
```

---

## Core syntax

**Indentation-based blocks** — Python uses consistent indentation (spaces) to mark what's
"inside" a block, instead of `{ }`.
```python
for item in items:
    print(item)        # inside the loop
    print("still inside")
print("done")           # outside the loop (less indented)
```
```csharp
foreach (var item in items)
{
    Console.WriteLine(item);   // inside the loop
}
Console.WriteLine("done");     // outside the loop
```

**`def function_name(params):`** — declares a function.
*C# equivalent:* a method, minus the required return type / access modifier.
```python
def add_numbers(a: int, b: int) -> int:
    return a + b
```
```csharp
public int AddNumbers(int a, int b)
{
    return a + b;
}
```
Differences: `def` replaces the C# return-type-first style; there's no access modifier
(`public`/`private`) on Python functions this way; parameter/return types (`: int`, `-> int`)
are optional *hints*, not enforced at runtime like C#'s required types; the block starts with
`:` + indentation instead of `{ }`.

**Docstrings (`"""..."""`)** — a string right under a `def` or at the top of a file,
documenting what it does. Tools/IDEs read these automatically.
*C# equivalent:* XML doc comments (`/// <summary>`).
```python
def extract_pages(pdf_path: Path) -> list[str]:
    """Read a PDF and return a list of raw text, one entry per page."""
    ...
```
```csharp
/// <summary>Read a PDF and return a list of raw text, one entry per page.</summary>
public List<string> ExtractPages(Path pdfPath) { ... }
```

**Type hints** (e.g. `def f(x: int) -> str:`) — optional annotations stating expected
parameter/return types. Python doesn't enforce them at runtime, but they document intent and
let editors/tools catch mistakes.
*C# equivalent:* normal C# type declarations, except Python's are optional and unenforced.
```python
def clean_pages(pages_text: list[str]) -> tuple[list[str], set[str]]:
    ...
```
```csharp
public (List<string>, HashSet<string>) CleanPages(List<string> pagesText) { ... }
```

**f-strings** (`f"File: {name}"`) — embed variables/expressions directly inside a string.
*C# equivalent:* string interpolation, `$"File: {name}"`.
```python
name = "1706.03762.pdf"
print(f"File: {name}")          # File: 1706.03762.pdf
```
```csharp
string name = "1706.03762.pdf";
Console.WriteLine($"File: {name}");   // File: 1706.03762.pdf
```

**f-string format specifiers** (`f"{chunk_index:04d}"`) — `:04d` means "format this integer
with at least 4 digits, zero-padded" (`7` → `0007`).
*C# equivalent:* `chunkIndex.ToString("D4")`.
```python
chunk_index = 7
print(f"{chunk_index:04d}")     # 0007
```
```csharp
int chunkIndex = 7;
Console.WriteLine(chunkIndex.ToString("D4"));   // 0007
```

**`{value!r}` in an f-string** — inserts the *repr* (debug representation) of a value instead
of its normal string form — e.g. shows quotes around strings, useful for spotting hidden
whitespace.
```python
line = "Page 3  "
print(f"{line!r}")     # 'Page 3  '   <- quotes + trailing spaces visible
print(f"{line}")       # Page 3      <- trailing spaces invisible to the eye
```
```csharp
// closest: use a debugger watch, or wrap manually
Console.WriteLine($"'{line}'");   // no true equivalent built into string interpolation
```

**`if __name__ == "__main__":`** — only run this code when the file is executed directly
(`python file.py`), not when another file imports functions from it.
*C# equivalent:* no exact match — C# always has a separate `Main()` entry point that never
runs just from referencing a DLL; this is Python's manual way of recreating that separation.
```python
def helper():
    return 42

if __name__ == "__main__":
    print(helper())      # only prints when you run: python this_file.py
```
```csharp
// C# already separates this via a dedicated entry point:
class Program
{
    static void Main()          // only this runs when the program starts
    {
        Console.WriteLine(Helper());
    }
    static int Helper() => 42;  // referencing this class elsewhere never runs Main()
}
```

**`continue`** — skips the rest of the current loop iteration, moves to the next one.
*C# equivalent:* same keyword, same behavior.
```python
for line in lines:
    if line.strip() == "":
        continue          # skip blank lines, don't process them further
    print(line)
```
```csharp
foreach (var line in lines)
{
    if (line.Trim() == "") continue;
    Console.WriteLine(line);
}
```

**Default parameter values** (`def f(threshold: float = 0.6):`) — if the caller doesn't
supply a value, this default is used.
*C# equivalent:* same concept, same syntax style (`float threshold = 0.6f`).
```python
def find_repeated_lines(pages_lines, threshold: float = 0.6):
    ...

find_repeated_lines(pages)          # uses threshold=0.6
find_repeated_lines(pages, 0.8)     # overrides it
```
```csharp
public HashSet<string> FindRepeatedLines(List<List<string>> pagesLines, float threshold = 0.6f)
{ ... }
```

**Global constants** (`TARGET_WORDS = 250`, ALL_CAPS naming) — Python convention for values
meant to be constant, declared at module level (top of the file, outside any function).
*C# equivalent:* `public const int TargetWords = 250;`
```python
TARGET_WORDS = 250   # top of chunking.py, outside any function
```
```csharp
public const int TargetWords = 250;
```

---

## Data structures used

**`list`** — an ordered, changeable collection, written `[1, 2, 3]`.
*C# equivalent:* `List<T>`.
```python
numbers = [1, 2, 3]
numbers.append(4)        # [1, 2, 3, 4]
```
```csharp
var numbers = new List<int> { 1, 2, 3 };
numbers.Add(4);          // [1, 2, 3, 4]
```

**`tuple`** — an ordered, *fixed* grouping of values, written `(1, "a")`. Used here for
`(page_number, paragraph_text)` pairs.
*C# equivalent:* closest is a `ValueTuple`, e.g. `(int, string)`.
```python
pair = (3, "Attention is all you need...")
page_number, paragraph_text = pair    # unpack it
```
```csharp
var pair = (3, "Attention is all you need...");
var (pageNumber, paragraphText) = pair;
```

**Returning multiple values** (`return cleaned_pages, repeated_lines`) — Python functions can
return more than one value at once as a tuple; the caller unpacks them:
`cleaned, removed = clean_pages(pages)`.
*C# equivalent:* returning a `ValueTuple`, or an `out` parameter.
```python
def clean_pages(pages_text):
    ...
    return cleaned_pages, repeated_lines

cleaned, removed = clean_pages(pages_text)
```
```csharp
public (List<string> Cleaned, HashSet<string> Removed) CleanPages(List<string> pagesText)
{
    ...
    return (cleanedPages, repeatedLines);
}
var (cleaned, removed) = CleanPages(pagesText);
```

**`set`** — an unordered collection with no duplicates, fast membership checks (`x in my_set`).
*C# equivalent:* `HashSet<T>`.
```python
seen = {"abstract", "introduction"}
seen.add("results")
print("abstract" in seen)   # True
```
```csharp
var seen = new HashSet<string> { "abstract", "introduction" };
seen.Add("results");
Console.WriteLine(seen.Contains("abstract"));   // True
```

**`dict`** — key-value pairs, written `{"key": "value"}`. Used for chunk metadata records.
*C# equivalent:* `Dictionary<TKey, TValue>`.
```python
chunk = {"chunk_id": "1706.03762-0007", "word_count": 287}
print(chunk["word_count"])   # 287
```
```csharp
var chunk = new Dictionary<string, object> { ["chunk_id"] = "1706.03762-0007", ["word_count"] = 287 };
Console.WriteLine(chunk["word_count"]);   // 287
```

**`collections.Counter`** — like a dictionary that automatically starts counts at 0 and
increments, no need to check "does this key exist yet."
*C# equivalent:* closest is a `Dictionary<string, int>` you'd otherwise have to manage
manually (`if (!dict.ContainsKey(k)) dict[k] = 0; dict[k]++;`) — `Counter` does that for you.
```python
from collections import Counter
counts = Counter()
counts["header line"] += 1
counts["header line"] += 1
print(counts["header line"])   # 2, no KeyError even though it was never explicitly set
```
```csharp
var counts = new Dictionary<string, int>();
if (!counts.ContainsKey("header line")) counts["header line"] = 0;
counts["header line"]++;
if (!counts.ContainsKey("header line")) counts["header line"] = 0;
counts["header line"]++;
Console.WriteLine(counts["header line"]);   // 2
```

**List/set comprehensions** — a compact way to build a new collection by transforming each
item of an existing one:
```python
pages_lines = [page.split("\n") for page in pages_text]

# equivalent to:
pages_lines = []
for page in pages_text:
    pages_lines.append(page.split("\n"))
```
*C# equivalent:* LINQ, e.g. `pagesText.Select(p => p.Split('\n')).ToList()`.
```csharp
var pagesLines = pagesText.Select(p => p.Split('\n')).ToList();
```

**Negative slicing** (`words[-45:]`) — takes the *last* 45 items of a list.
*C# equivalent:* no direct syntax; closest is `words.Skip(words.Count - 45)`.
```python
words = ["a", "b", "c", "d", "e"]
print(words[-2:])   # ['d', 'e']  <- last 2 items
```
```csharp
var words = new List<string> { "a", "b", "c", "d", "e" };
var lastTwo = words.Skip(words.Count - 2).ToList();   // ["d", "e"]
```

**Tuple unpacking with `_`** (`for page_num, _ in items:`) — `_` is a convention meaning "I
don't need this value, just ignoring it."
```python
items = [(1, "intro"), (2, "background")]
for page_num, _ in items:
    print(page_num)    # 1, 2  <- the text is unpacked but never used
```
```csharp
// C# lets you just not unpack the part you don't need:
foreach (var (pageNum, _) in items)
    Console.WriteLine(pageNum);   // C# 8+ also supports the "_" discard pattern
```

---

## Built-in functions used

**`len(x)`** — length of a string/list/etc. *C# equivalent:* `.Length` / `.Count`.
```python
print(len("hello"))       # 5
print(len([1, 2, 3]))     # 3
```
```csharp
Console.WriteLine("hello".Length);           // 5
Console.WriteLine(new List<int>{1,2,3}.Count); // 3
```

**`sum(iterable)`**, **`min(iterable)`**, **`max(iterable)`** — add up / find smallest / find
largest across a collection. *C# equivalent:* LINQ's `.Sum()`, `.Min()`, `.Max()`.
```python
word_counts = [287, 190, 305]
print(sum(word_counts), min(word_counts), max(word_counts))   # 782 190 305
```
```csharp
var wordCounts = new List<int> { 287, 190, 305 };
Console.WriteLine($"{wordCounts.Sum()} {wordCounts.Min()} {wordCounts.Max()}");
```

**`enumerate(iterable, start=1)`** — loop over a collection while also getting an index/counter
alongside each item, starting the count at 1 instead of the default 0.
*C# equivalent:* `list.Select((item, i) => (i, item))`.
```python
for page_number, page_text in enumerate(pages_text, start=1):
    print(page_number, page_text[:20])   # 1 ..., 2 ..., 3 ...
```
```csharp
foreach (var (pageNumber, pageText) in pagesText.Select((text, i) => (i + 1, text)))
    Console.WriteLine($"{pageNumber} {pageText.Substring(0, 20)}");
```

**`open(path, mode, encoding=...)` combined with `with ... as f:`** — opens a file. The
`with` block (a "context manager") automatically closes the file when the block ends, even if
an error happens inside it.
*C# equivalent:* `using (var f = new StreamWriter(path)) { ... }`.
```python
with open("data/processed/chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)
# file is automatically closed here, even if json.dump raised an error
```
```csharp
using (var f = new StreamWriter("data/processed/chunks.json"))
{
    f.Write(JsonSerializer.Serialize(allChunks));
}   // file is automatically closed here too
```

---

## String methods used

**`.strip()`** — removes leading/trailing whitespace. *C# equivalent:* `.Trim()`.
```python
print("  hello  ".strip())    # "hello"
```
```csharp
Console.WriteLine("  hello  ".Trim());   // "hello"
```

**`.split(separator)`** — breaks a string into a list of pieces at each occurrence of
`separator` (e.g. `"\n"` for lines, `"\n\n"` for paragraphs).
*C# equivalent:* `.Split(...)`.
```python
"one\ntwo\nthree".split("\n")     # ['one', 'two', 'three']
```
```csharp
"one\ntwo\nthree".Split('\n');    // ["one", "two", "three"]
```

**`.splitlines()`** — splits a string into a list of lines, without needing to specify `"\n"`
explicitly (also handles different line-ending styles).
```python
"line1\nline2\r\nline3".splitlines()   # ['line1', 'line2', 'line3']
```
```csharp
text.Split(new[] { "\r\n", "\n" }, StringSplitOptions.None);
```

**`"\n".join(list_of_strings)`** — glues a list of strings back together, inserting `"\n"`
between each. *C# equivalent:* `string.Join("\n", list)`.
```python
"\n".join(["one", "two", "three"])    # "one\ntwo\nthree"
```
```csharp
string.Join("\n", new[] { "one", "two", "three" });   // "one\ntwo\nthree"
```

**`in` (membership test)** — `if x in my_set:` checks whether `x` is present.
*C# equivalent:* `HashSet<T>.Contains(x)`.
```python
if "abstract" in seen:
    print("already seen")
```
```csharp
if (seen.Contains("abstract"))
    Console.WriteLine("already seen");
```

---

## Regex (the `re` module)

*C# equivalent throughout:* `System.Text.RegularExpressions`.

**`re.compile(pattern, flags=...)`** — pre-parses a regex pattern once for reuse, instead of
Python re-parsing the pattern string every time. *C# equivalent:* caching a `Regex` object
instead of creating one from a string on every call.
```python
import re
SECTION_PATTERN = re.compile(r"^(introduction|results)", re.IGNORECASE)
```
```csharp
using System.Text.RegularExpressions;
var sectionPattern = new Regex(@"^(introduction|results)", RegexOptions.IgnoreCase);
```

**`re.fullmatch(pattern, text)`** — checks if the **entire** string matches the pattern.
*C# equivalent:* `Regex.IsMatch(text, "^" + pattern + "$")`.
```python
re.fullmatch(r"\d+", "42")      # matches (True-ish)
re.fullmatch(r"\d+", "42 of 3") # None (doesn't match — extra text)
```
```csharp
Regex.IsMatch("42", @"^\d+$");        // True
Regex.IsMatch("42 of 3", @"^\d+$");   // False
```

**`.match(text)`** (on a compiled pattern) — checks if the **start** of the string matches
(doesn't need to match the whole thing, unlike `fullmatch`).
```python
SECTION_PATTERN.match("Introduction to Transformers")   # matches — starts with "Introduction"
```
```csharp
sectionPattern.IsMatch("Introduction to Transformers");   // similar, though .NET's IsMatch scans anywhere unless anchored with ^
```

**`re.sub(pattern, replacement, text)`** — find-and-replace using a regex pattern.
*C# equivalent:* `Regex.Replace(...)`.
```python
re.sub(r"\n{3,}", "\n\n", "a\n\n\n\nb")   # "a\n\nb"
```
```csharp
Regex.Replace("a\n\n\n\nb", @"\n{3,}", "\n\n");   // "a\n\nb"
```

**`flags=re.IGNORECASE`** — makes the pattern match regardless of upper/lower case.
```python
re.fullmatch(r"page \d+", "PAGE 3", flags=re.IGNORECASE)   # matches
```
```csharp
Regex.IsMatch("PAGE 3", @"^page \d+$", RegexOptions.IgnoreCase);   // true
```

---

## Files & paths

**`pathlib.Path`** — represents a file/folder path in a cross-platform way (works the same on
Windows/Linux). *C# equivalent:* closest is combining `System.IO.Path` + `FileInfo`.
```python
from pathlib import Path
folder = Path("data/pdfs")
print(folder / "1706.03762.pdf")   # data/pdfs/1706.03762.pdf
```
```csharp
string folder = "data/pdfs";
string filePath = Path.Combine(folder, "1706.03762.pdf");
```

**`Path.glob("*.pdf")`** — finds all files in a folder matching a pattern (here, all `.pdf`
files). *C# equivalent:* `Directory.GetFiles(path, "*.pdf")`.
```python
for pdf_file in Path("data/pdfs").glob("*.pdf"):
    print(pdf_file.name)
```
```csharp
foreach (var pdfFile in Directory.GetFiles("data/pdfs", "*.pdf"))
    Console.WriteLine(Path.GetFileName(pdfFile));
```

**`Path.mkdir(parents=True, exist_ok=True)`** — creates a folder. `parents=True` also creates
any missing parent folders; `exist_ok=True` means "don't error if it already exists."
*C# equivalent:* `Directory.CreateDirectory(...)` (already safe to call if the folder exists).
```python
Path("data/processed").mkdir(parents=True, exist_ok=True)
```
```csharp
Directory.CreateDirectory("data/processed");   // safe even if it already exists
```

**`json.dump(obj, file, indent=2)`** — writes a Python object (list/dict) out as a JSON file.
`indent=2` just makes the output human-readable/pretty-printed.
*C# equivalent:* `System.Text.Json.JsonSerializer.Serialize(...)` written to a file.
```python
import json
with open("data/processed/chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2)
```
```csharp
using System.Text.Json;
File.WriteAllText("data/processed/chunks.json",
    JsonSerializer.Serialize(allChunks, new JsonSerializerOptions { WriteIndented = true }));
```

---

---

## More built-ins used (vectorstore.py)

**`zip(list1, list2)`** — pairs up two lists element-by-element, so you can loop over both at
once. *C# equivalent:* `list1.Zip(list2, (a, b) => (a, b))`.
```python
scores = [0.91, 0.85]
rows = [12, 47]
for score, row in zip(scores, rows):
    print(score, row)     # 0.91 12   then   0.85 47
```
```csharp
var scores = new List<double> { 0.91, 0.85 };
var rows = new List<int> { 12, 47 };
foreach (var (score, row) in scores.Zip(rows, (s, r) => (s, r)))
    Console.WriteLine($"{score} {row}");
```

**`range(start, stop, step)`** — generates a sequence of numbers to loop over, without
building a real list in memory. *C# equivalent:* a plain `for` loop with a counter.
```python
for start in range(0, 100, 32):
    print(start)     # 0, 32, 64, 96  <- batch starting positions
```
```csharp
for (int start = 0; start < 100; start += 32)
    Console.WriteLine(start);
```

**`assert condition, "message"`** — crashes the program immediately with the given message if
`condition` is false; does nothing if it's true. Used as a quick internal sanity check, not for
validating user input. *C# equivalent:* `Debug.Assert(condition, "message")` or `Trace.Assert`.
```python
assert reloaded_index.ntotal == len(chunks), "index size doesn't match chunk count!"
```
```csharp
Debug.Assert(reloadedIndex.Total == chunks.Count, "index size doesn't match chunk count!");
```

---

## HTTP requests (the `requests` library)

*C# equivalent throughout:* `HttpClient`.

**`requests.post(url, json=..., timeout=...)`** — sends an HTTP POST request with a JSON body.
`timeout` is in seconds; the call raises an exception if the server doesn't respond in time.
```python
response = requests.post(
    "http://localhost:11434/api/embed",
    json={"model": "nomic-embed-text", "input": ["hello", "world"]},
    timeout=120,
)
```
```csharp
var client = new HttpClient { Timeout = TimeSpan.FromSeconds(120) };
var content = new StringContent(
    JsonSerializer.Serialize(new { model = "nomic-embed-text", input = new[] { "hello", "world" } }),
    Encoding.UTF8, "application/json");
var response = await client.PostAsync("http://localhost:11434/api/embed", content);
```

**`response.raise_for_status()`** — throws an exception if the HTTP status code indicates an
error (4xx/5xx), does nothing on success. Lets you fail fast instead of silently continuing
with a bad response. *C# equivalent:* `response.EnsureSuccessStatusCode()`.

**`response.json()`** — parses the response body as JSON and returns it as Python
dicts/lists. *C# equivalent:* `await response.Content.ReadFromJsonAsync<T>()`.

**`try: ... except requests.exceptions.RequestException as error: ...`** — Python's version of
a try/catch block; catches network-related failures (timeouts, connection errors, bad status
from `raise_for_status()`) so the code can retry instead of crashing immediately.
*C# equivalent:* `try { ... } catch (HttpRequestException error) { ... }`.
```python
try:
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
except requests.exceptions.RequestException as error:
    print(f"failed: {error}")
```
```csharp
try
{
    var response = await client.PostAsync(url, content);
    response.EnsureSuccessStatusCode();
}
catch (HttpRequestException error)
{
    Console.WriteLine($"failed: {error.Message}");
}
```

---

## NumPy basics (the `numpy` library, imported as `np`)

*C# equivalent throughout:* closest is `System.Numerics` / a math library like Math.NET —
NumPy's arrays are like a strongly-typed, vectorized `double[,]` with built-in math operations.

**`np.array(list, dtype="float32")`** — converts a plain Python list (or list of lists) into a
NumPy array, a fixed-type, math-optimized alternative to `list`.
```python
vectors = np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32")   # shape (2, 2)
```

**`.shape`** — the dimensions of an array, e.g. `(398, 768)` = 398 rows (one per chunk), 768
columns (the embedding's dimension count). *C# equivalent:* `array.GetLength(0)`, `.GetLength(1)`.

**`np.linalg.norm(vectors, axis=1, keepdims=True)`** — computes the length (magnitude) of
each vector (row) in the array. `axis=1` means "compute per row, not for the whole array at
once"; `keepdims=True` keeps the result shaped so it can be divided back into the original
array directly (shape `(398, 1)` instead of flattening to `(398,)`).
```python
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
unit_vectors = vectors / norms     # every row now has length 1.0
```

**`np.save(path, array)`** / **`np.load(path)`** — writes/reads a NumPy array to/from a `.npy`
binary file — much faster and more compact than saving as JSON/text for large numeric data.

---

## FAISS basics (the `faiss` library)

FAISS has no direct C# equivalent in common use — closest conceptual comparison is a
specialized search index/database engine, but accessed as an in-process library, not a
server you connect to over a network (unlike Ollama).

**`faiss.IndexFlatIP(dimension)`** — creates a new, empty index that does exact ("Flat")
nearest-neighbor search using inner product ("IP") as the similarity measure. `dimension` must
match the length of the vectors you'll add (768 for `nomic-embed-text`).
```python
index = faiss.IndexFlatIP(768)
```

**`index.add(vectors)`** — adds a 2D array of vectors to the index, one row = one entry. Row
order becomes each entry's row number (its retrieval "ID").
```python
index.add(embeddings)    # embeddings.shape == (398, 768) -> rows 0-397 added
```

**`index.search(query_vector, k)`** — finds the `k` closest vectors to `query_vector`. Returns
two arrays: similarity scores, and the row numbers of the matches (both shaped
`(number_of_queries, k)`, since you could search multiple queries at once — here always just 1).
```python
scores, row_indices = index.search(query_vector, k=3)
# scores[0]      -> [0.91, 0.85, 0.80]   (top-3 similarity scores for our 1 query)
# row_indices[0] -> [12, 47, 203]        (which rows in the index those were)
```

**`faiss.write_index(index, path)`** / **`faiss.read_index(path)`** — save/load an index
to/from a file on disk, so it doesn't need to be rebuilt (re-embedding every chunk) every time
the program runs.

---

---

## Classes (bot.py)

**`class ChatSession:`** — declares a class, a blueprint for creating objects that bundle
data (state) together with functions that act on that data. *C# equivalent:* `class ChatSession { }`.
```python
class ChatSession:
    def __init__(self, chunks, index):
        self.chunks = chunks
        self.history = deque(maxlen=4)
```
```csharp
public class ChatSession
{
    private List<Dictionary<string, object>> chunks;
    private Queue<Turn> history = new Queue<Turn>(4);

    public ChatSession(List<Dictionary<string, object>> chunks)
    {
        this.chunks = chunks;
    }
}
```

**`def __init__(self, ...):`** — the constructor: runs automatically when you create a new
object from the class, used to set up its initial state.
*C# equivalent:* a constructor, `public ChatSession(...) { }`.
```python
session = ChatSession(chunks, index)   # __init__ runs automatically here
```
```csharp
var session = new ChatSession(chunks, index);   // constructor runs automatically here
```

**`self`** — inside a class's methods, `self` refers to "this particular object" — how a
method reads/writes the object's own data. *C# equivalent:* `this` (except C# lets you omit
it in most cases; Python requires `self` as an explicit first parameter on every method).
```python
class ChatSession:
    def ask(self, question):
        self.history.append(question)   # "this object's" history
```
```csharp
public class ChatSession
{
    public void Ask(string question)
    {
        this.history.Add(question);   // "this" is often omittable in C#, unlike Python's self
    }
}
```

**`session.ask("...")`** — calling a method on an object. *C# equivalent:* `session.Ask("...")`.

---

## `collections.deque` (a bounded queue)

**`deque(maxlen=4)`** — a list-like collection that automatically drops the oldest item once
more than `maxlen` items have been added. Used here to hold exactly the last 4 conversation
turns, with zero manual "remove the oldest if too many" logic needed.
*C# equivalent:* no exact built-in match — closest is manually managing a `Queue<T>` and
calling `Dequeue()` yourself whenever `Count > 4`.
```python
from collections import deque
history = deque(maxlen=4)
for i in range(1, 6):
    history.append(f"turn {i}")
print(list(history))   # ['turn 2', 'turn 3', 'turn 4', 'turn 5']  <- turn 1 auto-dropped
```
```csharp
var history = new Queue<string>();
for (int i = 1; i <= 5; i++)
{
    history.Enqueue($"turn {i}");
    if (history.Count > 4) history.Dequeue();   // manual eviction -- deque does this for you
}
```

---

## Interactive input

**`input("prompt text")`** — pauses the program and waits for the user to type something in
the terminal, then returns what they typed as a string.
*C# equivalent:* `Console.ReadLine()` (though C#'s doesn't show a prompt string itself — you'd
`Console.Write("prompt text")` first).
```python
question = input("You: ").strip()
```
```csharp
Console.Write("You: ");
string question = Console.ReadLine().Trim();
```

**`while True: ... if condition: break`** — an infinite loop that only stops when `break`
runs, used here to keep the chat session going until the user types "quit".
*C# equivalent:* `while (true) { ... if (condition) break; }` — identical concept and keyword.

---

## LangChain patterns (`UseLangchain` branch — langchain_service.py)

**Constructing an object with only keyword arguments** — Python lets you (and these libraries
often require you to) pass constructor arguments strictly by name, in any order, rather than
positionally.
*C# equivalent:* using named arguments explicitly, e.g. `new ChatOllama(model: "...", temperature: 0)`.
```python
llm = ChatOllama(model="llama3.2", temperature=0.0, num_ctx=8192)
```

**`**kwargs` — unpacking a dict into keyword arguments.** Build the arguments as a dict first
(so you can conditionally add to it), then splat it into the constructor with `**`.
*C# equivalent:* no direct match — closest is building an options/config object and passing it,
since C# can't expand a dictionary into named parameters at a call site.
```python
kwargs = {"model": CHAT_MODEL, "temperature": temperature}
if json_mode:
    kwargs["format"] = "json"      # conditionally add an argument
llm = ChatOllama(**kwargs)          # ** expands the dict into named arguments
```

**A module-level cache + the `global` keyword.** Variables assigned inside a function are local
by default; `global` says "assign to the module-level one instead." Used here so an expensive
object is built once and reused.
*C# equivalent:* a `static` field with lazy initialization, or `Lazy<T>`.
```python
_EMBEDDINGS = None

def get_embeddings():
    global _EMBEDDINGS               # without this, the assignment below
    if _EMBEDDINGS is None:          # would create a new *local* variable
        _EMBEDDINGS = OllamaEmbeddings(model=EMBED_MODEL)
    return _EMBEDDINGS
```
```csharp
private static OllamaEmbeddings _embeddings;
public static OllamaEmbeddings GetEmbeddings() =>
    _embeddings ??= new OllamaEmbeddings(EmbedModel);
```

**Using a tuple as a dictionary key** — Python tuples are hashable, so a combination of values
can key a cache directly, no composite-key class needed.
*C# equivalent:* `Dictionary<(double, bool), ChatOllama>` — C# value tuples work the same way.
```python
_LLM_CACHE = {}
key = (temperature, json_mode)       # a 2-value tuple as the key
if key not in _LLM_CACHE:
    _LLM_CACHE[key] = ChatOllama(...)
```

**Operator overloading — the `|` in LangChain (LCEL).** `|` is normally bitwise-OR, but Python
lets a class redefine what operators mean for it. LangChain uses this to compose steps into a
pipeline, left to right.
*C# equivalent:* C# also supports operator overloading (`public static X operator |(...)`),
though chaining pipelines this way is far less common than LINQ-style method chaining.
```python
chain = prompt | llm | StrOutputParser()   # output of each step feeds the next
answer = chain.invoke({"question": "...", "context": "..."})
```

**Union type hints with `|`** (`OllamaEmbeddings | None`) — declares a value that may be either
that type or `None`. Same `|` symbol, unrelated to the operator overloading above.
*C# equivalent:* a nullable reference type, `OllamaEmbeddings?`.
```python
_EMBEDDINGS: OllamaEmbeddings | None = None
```

**List slicing to split a batch** — `list[:n]` takes the first n items, `list[n:]` takes
everything after them. Neither modifies the original.
*C# equivalent:* `list.Take(n)` and `list.Skip(n)`.
```python
first, rest = documents[:batch_size], documents[batch_size:]
```
```csharp
var first = documents.Take(batchSize).ToList();
var rest  = documents.Skip(batchSize).ToList();
```

---

*(This file will keep growing as later stages — evaluation — introduce new libraries and
concepts.)*
