from pydantic import BaseModel

params_file_name = './params.json'


class Params(BaseModel):
    max_n_iters: int
    n_shots: int
    noise_sigma: float
    gamma1: float
    gamma2: float
    parent: str | None


def main():
    pass


if __name__ == '__main__':
    main()