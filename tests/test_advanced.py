from rag_intelligence.advanced import chunk_text, SemanticIndex, grounded_prompt
def test_semantic_index():
    embed=lambda texts:[[float("cat" in x.lower()),float("dog" in x.lower())] for x in texts]
    idx=SemanticIndex(embed)
    idx.add(chunk_text("pets.txt","Cat is a pet."))
    hits=idx.query("cat")
    assert hits[0]["source"]=="pets.txt"
    assert "[1]" in grounded_prompt("What?",hits)
