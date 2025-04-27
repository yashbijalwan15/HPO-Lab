from hpo_algorithm import HPOAlgorithm


class RandomSearch(HPOAlgorithm):
    def ask(self) -> tuple[dict, float]:
        pass
    
    def tell(self, config, result, bugdet) -> None:
        pass