from molmo_spaces.molmo_spaces_constants import get_scenes
mapping = get_scenes("procthor-objaverse", "train")
keys = list(mapping["train"].keys())
print("Total keys:", len(keys))
print("First 40 keys:", keys[:40])
