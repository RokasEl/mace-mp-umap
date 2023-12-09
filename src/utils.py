def get_group_and_period(symbol):
    if symbol not in ELEMENT_DICT:
        element = mendeleev.element(symbol)
        group = element.group_id
        group = -1 if group is None else group
        ELEMENT_DICT[symbol] = (group, element.period)
    return ELEMENT_DICT[symbol]
