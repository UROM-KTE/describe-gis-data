def string_list_generator(initial_string: str = 'sample', length: int = 10) -> list[str]:
    return [initial_string + str(i) for i in range(length)]
