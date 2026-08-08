#!/bin/bash

# ============================================================================
# Sampled Corpus Summarizer
#
# Recursively summarizes .txt files using representative samples rather than
# processing every line of every document.
#
# For large files, the sample consists of:
#
#     first  2000 lines
#     middle 1000 lines
#     final  1000 lines
#
# The three regions are combined into a single prompt and summarized by
# Granite. All summaries are appended to one consolidated output file.
#
# Progress is recorded so completed files are skipped on subsequent runs.
# ============================================================================


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

progress_file="progress.log"
summary_file="axial-paradigms.txt"

main_dir=$(pwd)

model="granite4.1:3b"

num_ctx=32768

start_lines=2000
middle_lines=1000
end_lines=1000

# If a file has fewer than this many lines, summarize the entire file.
whole_file_threshold=5000

# Conservative character ceiling.
#
# Our benchmark on the current corpus measured roughly 3.1 characters/token.
# 70,000 characters therefore corresponds to roughly 22,500 tokens, leaving
# substantial room inside the 32,768-token context for instructions and
# generated output.
max_sample_chars=70000


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

touch "$main_dir/$progress_file"
touch "$main_dir/$summary_file"


# ---------------------------------------------------------------------------
# Progress handling
# ---------------------------------------------------------------------------

is_processed() {
    grep -Fxq "$1" "$main_dir/$progress_file"
}


echo "Script started at $(date)" \
    >> "$main_dir/$progress_file"

echo "Model: $model" \
    >> "$main_dir/$progress_file"

echo "Context: $num_ctx" \
    >> "$main_dir/$progress_file"

echo "Summaries will be saved to $summary_file" \
    >> "$main_dir/$progress_file"


# ---------------------------------------------------------------------------
# Construct representative sample
# ---------------------------------------------------------------------------

make_sample() {

    local file="$1"
    local output="$2"

    local total_lines

    total_lines=$(wc -l < "$file")

    if [ "$total_lines" -le "$whole_file_threshold" ]; then

        {
            echo "============================================================"
            echo "COMPLETE DOCUMENT"
            echo "Total lines: $total_lines"
            echo "============================================================"
            echo

            cat "$file"

        } > "$output"

        return
    fi


    local middle_start
    local middle_end

    middle_start=$(( total_lines / 2 - middle_lines / 2 + 1 ))

    if [ "$middle_start" -lt 1 ]; then
        middle_start=1
    fi

    middle_end=$(( middle_start + middle_lines - 1 ))


    {
        echo "============================================================"
        echo "BEGINNING SAMPLE"
        echo "Lines 1-$start_lines of $total_lines"
        echo "============================================================"
        echo

        head -n "$start_lines" "$file"

        echo
        echo
        echo "============================================================"
        echo "MIDDLE SAMPLE"
        echo "Lines $middle_start-$middle_end of $total_lines"
        echo "============================================================"
        echo

        sed -n "${middle_start},${middle_end}p" "$file"

        echo
        echo
        echo "============================================================"
        echo "ENDING SAMPLE"
        echo "Final $end_lines lines of $total_lines"
        echo "============================================================"
        echo

        tail -n "$end_lines" "$file"

    } > "$output"
}


# ---------------------------------------------------------------------------
# Process one file
# ---------------------------------------------------------------------------

process_file() {

    local file="$1"

    local file_name
    local title
    local total_lines
    local sample_file
    local sample_chars

    file_name=$(basename "$file")
    title=$(basename "$file" .txt)


    # Never summarize our own accumulated output.
    if [ "$file_name" = "$summary_file" ]; then
        echo "Skipping summary file: $summary_file"
        return
    fi


    # Never treat the progress log as source material.
    if [ "$file_name" = "$progress_file" ]; then
        return
    fi


    # Skip files already completed.
    if is_processed "$file_name"; then
        echo "Already processed: $file_name"
        return
    fi


    total_lines=$(wc -l < "$file")


    echo
    echo "============================================================"
    echo "Processing: $file_name"
    echo "Lines:      $total_lines"
    echo "============================================================"


    echo "Processing $file_name" \
        >> "$main_dir/$progress_file"


    # -----------------------------------------------------------------------
    # Temporary sample
    # -----------------------------------------------------------------------

    sample_file=$(mktemp)

    make_sample "$file" "$sample_file"

    sample_chars=$(wc -c < "$sample_file")


    echo "Sample size: $sample_chars characters"


    # -----------------------------------------------------------------------
    # Protect the 32K context from pathological long-line documents.
    # -----------------------------------------------------------------------

    if [ "$sample_chars" -gt "$max_sample_chars" ]; then

        echo \
            "Sample exceeds ${max_sample_chars} characters; truncating."

        head -c "$max_sample_chars" "$sample_file" \
            > "${sample_file}.trimmed"

        mv "${sample_file}.trimmed" "$sample_file"

        sample_chars=$max_sample_chars

    fi


    # -----------------------------------------------------------------------
    # Add document header to consolidated summary
    # -----------------------------------------------------------------------

    {
        echo
        echo
        echo "================================================================"
        echo
        echo "### $title"
        echo
        echo "Source: $file"
        echo
        echo "Total lines: $total_lines"
        echo
        echo "Sampling: first $start_lines / middle $middle_lines / final $end_lines lines"
        echo
        echo "Sample characters: $sample_chars"
        echo

    } >> "$main_dir/$summary_file"


    # -----------------------------------------------------------------------
    # Summarize
    # -----------------------------------------------------------------------

    echo "Summarizing with $model..."


    ollama run "$model" \
        --nowordwrap \
        --verbose \
        "The following material consists of representative samples from a
longer document. The samples come from the beginning, middle, and end.

Produce a detailed, coherent summary of the document.

Identify its central subject, major arguments, important concepts, methods,
examples, and conclusions. Explain how the ideas appear to develop across
the document.

Do not assume that material absent from the samples was present in the
original. Distinguish direct evidence from reasonable inference where
necessary.

Do not merely describe the samples. Synthesize them into an account of the
document as a whole.

DOCUMENT SAMPLES:

$(cat "$sample_file")" \
        | tee -a "$main_dir/$summary_file"


    echo >> "$main_dir/$summary_file"


    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    rm -f "$sample_file"


    # -----------------------------------------------------------------------
    # Mark successful completion
    # -----------------------------------------------------------------------

    echo "$file_name" \
        >> "$main_dir/$progress_file"

    echo "Completed: $file_name"
}


# ---------------------------------------------------------------------------
# Process files in one directory
# ---------------------------------------------------------------------------

process_files() {

    local dir="$1"

    echo "Processing directory: $dir"


    for file in "$dir"/*.txt; do

        [ -e "$file" ] || continue
        [ -f "$file" ] || continue

        process_file "$file"

    done
}


# ---------------------------------------------------------------------------
# Recursive directory traversal
# ---------------------------------------------------------------------------

process_subdirectories() {

    local parent_dir="$1"


    for dir in "$parent_dir"/*/; do

        [ -d "$dir" ] || continue

        process_files "$dir"

        process_subdirectories "$dir"

    done
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

process_files "$main_dir"

process_subdirectories "$main_dir"


echo "Script completed at $(date)" \
    >> "$main_dir/$progress_file"


echo
echo "============================================================"
echo "SUMMARY RUN COMPLETE"
echo "============================================================"
echo
echo "Output:   $main_dir/$summary_file"
echo "Progress: $main_dir/$progress_file"

