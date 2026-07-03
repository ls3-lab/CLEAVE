from retriever_attack import git_sha


def test_git_sha_is_string():
    s = git_sha()
    assert isinstance(s, str) and (len(s) == 40 or s == "unknown")
