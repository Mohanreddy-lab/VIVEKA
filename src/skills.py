"""
skills.py — Skill synonym expansion and word-boundary matching.

Without this, a candidate who writes "PySpark" scores 0 for a JD that
says "Apache Spark". That's wrong. This file fixes it everywhere.
"""

import re

# Canonical name -> list of all equivalent surface forms (lowercased)
SKILL_SYNONYMS: dict[str, list[str]] = {
    # Python
    "python":               ["py", "python3", "python2", "cpython"],
    "pandas":               ["pd", "pandas dataframe"],
    "numpy":                ["np"],
    "fastapi":              ["fast api"],

    # Spark / streaming
    "apache spark":         ["spark", "pyspark", "spark sql", "sparksql",
                             "spark streaming", "structured streaming"],
    "apache kafka":         ["kafka", "confluent kafka", "kafka streams"],
    "apache flink":         ["flink"],
    "apache airflow":       ["airflow", "apache-airflow", "workflow orchestration"],
    "apache hive":          ["hive", "hiveql"],
    "apache hadoop":        ["hadoop", "hdfs", "mapreduce", "hbase"],
    "apache beam":          ["beam", "dataflow"],

    # Cloud — AWS
    "aws":                  ["amazon web services", "amazon aws", "amazon cloud"],
    "s3":                   ["amazon s3", "aws s3"],
    "redshift":             ["amazon redshift", "aws redshift"],
    "glue":                 ["aws glue", "amazon glue"],
    "lambda":               ["aws lambda", "amazon lambda", "serverless"],
    "ec2":                  ["amazon ec2"],
    "emr":                  ["amazon emr", "aws emr", "elastic mapreduce"],
    "kinesis":              ["aws kinesis", "amazon kinesis"],

    # Cloud — GCP
    "gcp":                  ["google cloud", "google cloud platform", "bigquery",
                             "dataflow", "dataproc", "cloud run"],

    # Cloud — Azure
    "azure":                ["microsoft azure", "ms azure", "azure databricks",
                             "azure data factory", "adf"],

    # Databases
    "sql":                  ["mysql", "postgresql", "postgres", "mssql", "t-sql",
                             "tsql", "pl/sql", "plsql", "ansi sql", "structured query"],
    "postgresql":           ["postgres", "pg", "pgresql"],
    "mysql":                ["my sql"],
    "mongodb":              ["mongo", "mongo db"],
    "elasticsearch":        ["elastic search", "elastic", "opensearch"],
    "redis":                ["redis cache", "redisearch"],
    "cassandra":            ["apache cassandra"],
    "snowflake":            ["snowflake data warehouse", "snowflake dwh"],
    "databricks":           ["delta lake", "delta tables", "lakehouse"],

    # Transformation / modeling
    "dbt":                  ["data build tool", "dbt core", "dbt cloud",
                             "dbt-core", "dbt-cloud"],
    "data modeling":        ["data modelling", "dimensional modeling", "dimensional modelling",
                             "star schema", "snowflake schema", "data model"],
    "data warehouse":       ["dwh", "edw", "data warehousing"],
    "etl":                  ["elt", "extract transform load", "data pipeline",
                             "data pipelines", "pipeline"],

    # Infrastructure / devops
    "docker":               ["dockerfile", "containerization", "containers", "container"],
    "kubernetes":           ["k8s", "container orchestration", "helm"],
    "terraform":            ["tf", "infrastructure as code", "iac", "terragrunt"],
    "git":                  ["github", "gitlab", "bitbucket", "version control", "vcs"],
    "ci/cd":                ["cicd", "ci cd", "continuous integration",
                             "continuous delivery", "continuous deployment",
                             "github actions", "jenkins", "gitlab ci"],
    "linux":                ["unix", "bash", "shell scripting", "shell script", "cli"],

    # ML / AI
    "machine learning":     ["ml", "deep learning", "neural networks", "ai",
                             "artificial intelligence"],
    "tensorflow":           ["keras", "tf"],
    "pytorch":              ["torch"],
    "scikit-learn":         ["sklearn", "scikit learn"],
    "mlflow":               ["ml flow"],
    "feature store":        ["feature stores", "feast"],

    # Languages
    "java":                 ["jvm", "spring boot", "spring", "maven", "gradle"],
    "scala":                ["akka", "sbt"],
    "javascript":           ["js", "node.js", "nodejs", "node js", "es6"],
    "typescript":           ["ts"],
    "go":                   ["golang"],
    "r":                    ["r language", "r programming", "rstats"],

    # Concepts
    "rest api":             ["rest", "restful", "http api", "api", "openapi", "swagger"],
    "microservices":        ["micro services", "service mesh", "soa"],
    "data governance":      ["data quality", "data catalog", "data lineage"],
    "orchestration":        ["workflow", "dag", "scheduler"],
}

# Build reverse map: every surface form -> canonical
_REVERSE: dict[str, str] = {}
for _canon, _syns in SKILL_SYNONYMS.items():
    _REVERSE[_canon.lower()] = _canon
    for _s in _syns:
        _REVERSE[_s.lower()] = _canon


def canonical(skill: str) -> str:
    """Return the canonical form of a skill, or the original lowercased."""
    return _REVERSE.get(skill.lower().strip(), skill.lower().strip())


def expand_skill(skill: str) -> set[str]:
    """Return all surface forms for a skill (skill + synonyms), lowercased."""
    canon = canonical(skill)
    result: set[str] = {skill.lower().strip(), canon}
    result.update(s.lower() for s in SKILL_SYNONYMS.get(canon, []))
    return result


def _strip_parens(skill: str) -> str:
    """'AWS (S3, Glue, Redshift)' -> 'AWS'. Handles JD skills with examples."""
    return re.sub(r'\s*\([^)]*\)', '', skill).strip()


def skill_matches(skill: str, candidate_text: str) -> bool:
    """True if the skill or any synonym appears as a word/phrase in candidate_text."""
    text = candidate_text.lower()
    # Also try the skill with parenthetical stripped (e.g. "AWS (S3, Glue)" -> "AWS")
    variants = expand_skill(skill) | expand_skill(_strip_parens(skill))
    for variant in variants:
        if variant and re.search(r'(?<!\w)' + re.escape(variant) + r'(?!\w)', text):
            return True
    return False


def matched_skills(jd_skills: list[str], candidate_text: str) -> tuple[list[str], list[str]]:
    """Return (matched, missing) lists for a set of JD skills against candidate text."""
    matched, missing = [], []
    for skill in jd_skills:
        (matched if skill_matches(skill, candidate_text) else missing).append(skill)
    return matched, missing
