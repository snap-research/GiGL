from examples.MAG240M.common import NUM_PAPER_FEATURES

QUERY_TEMPLATE_REINDEX_AUTHOR_WRITES_PAPER_TABLE = """
-- Firstly, we reindex the author to the same node space as papers
-- TOTAL_NUM_PAPERS as defined in https://ogb.stanford.edu/docs/lsc/mag240m/
-- The paper node ids are thus: 0 to 121751665; and the author node ids will now start from 121751666
SELECT
    author + {TOTAL_NUM_PAPERS} AS src,
    paper as dst
FROM
    `{author_writes_paper_table}`
"""

QUERY_TEMPLATE_CAST_TO_HOMOGENEOUS_EDGE_TABLE = """
-- Combine the paper cites paper, and the re-indexed author writes paper tables into a single edge table
SELECT
  src,
  dst
FROM
  `{reindexed_author_writes_paper_table}`
UNION ALL
SELECT
  src,
  dst
FROM
  `{paper_cites_paper_table}`
"""

QUERY_TEMPLATE_COMPUTED_NODE_DEGREE_TABLE = """
SELECT
  node_id,
  COUNT(*) AS degree
FROM (
  SELECT
    src AS node_id
  FROM
    `{homogeneous_edge_table}`
  UNION ALL
  SELECT
    dst AS node_id
  FROM
    `{homogeneous_edge_table}`
)
GROUP BY
  node_id
"""

QUERY_TEMPLATE_CAST_TO_INTERMEDIARY_HOMOGENEOUS_NODE_TABLE = (
    """
WITH authors AS (
    SELECT 
        DISTINCT src as author_id
    FROM 
        `{reindexed_author_writes_paper_table}`
)
SELECT 
    author_id as node_id,
"""
    + ",\n".join([f"    0 AS feat_{i}" for i in range(NUM_PAPER_FEATURES)])
    + """
FROM authors
UNION ALL
SELECT 
    paper as node_id,
"""
    + ",\n".join([f"    feat_{i}" for i in range(NUM_PAPER_FEATURES)])
    + """
FROM
    `{paper_table}`
"""
)


QUERY_TEMPLATE_GENERATE_HOMOGENEOUS_NODE_TABLE = (
    """
SELECT
    interim_node_table.node_id as node_id,
    node_degree_table.degree as degree,
"""
    + ",\n".join(
        [
            f"    interim_node_table.feat_{i} as feat_{i}"
            for i in range(NUM_PAPER_FEATURES)
        ]
    )
    + """
FROM
    `{interim_node_table}` AS interim_node_table
JOIN
    `{node_degree_table}` AS node_degree_table
ON
    interim_node_table.node_id = node_degree_table.node_id
"""
)
