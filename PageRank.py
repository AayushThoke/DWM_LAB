# ------------------------------------------------------------
# Implementation of PageRank Algorithm
# Using basic Python (no external libraries)
# ------------------------------------------------------------

# --- Step 1: Represent the web as a directed graph ---
# Each key = page, value = list of pages it links to
web_graph = {
    'A': ['B', 'C'],
    'B': ['C', 'D'],
    'C': ['A'],
    'D': ['C']
}

# --- Step 2: Initialize parameters ---
damping_factor = 0.85   # probability that user follows a link
num_iterations = 100     # number of iterations
num_pages = len(web_graph)

# initialize rank of each page equally
ranks = {page: 1 / num_pages for page in web_graph}

# --- Step 3: Iteratively calculate PageRank ---
for i in range(num_iterations):
    new_ranks = {}
    for page in web_graph:
        # basic PageRank formula:
        # PR(A) = (1 - d)/N + d * Σ(PR(i) / L(i))
        inbound_sum = 0
        for other_page, links in web_graph.items():
            if page in links:
                inbound_sum += ranks[other_page] / len(links)

        new_ranks[page] = (1 - damping_factor) / num_pages + damping_factor * inbound_sum

    ranks = new_ranks

# --- Step 4: Display final PageRank values ---
print("Final PageRank after", num_iterations, "iterations:\n")
for page, rank in ranks.items():
    print(f"Page {page}: {rank:.4f}")
