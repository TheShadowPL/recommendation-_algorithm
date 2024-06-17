from datetime import datetime
import numpy as np


def ocena_podobienstwa_artykułu(tagi_użytkownika, tagi_artykułu, encoded_tags, X_stopnia_podobienstwa):
    suma_ocen = 0

    for tag_art in tagi_artykułu:
        for tag_uz in tagi_użytkownika:
            tag_art_vec = encoded_tags[tag_art]
            tag_uz_vec = encoded_tags[tag_uz]
            if tag_art_vec is not None and tag_uz_vec is not None:
                similarity = np.dot(tag_art_vec, tag_uz_vec) / (
                            np.linalg.norm(tag_art_vec) * np.linalg.norm(tag_uz_vec))
                if similarity >= X_stopnia_podobienstwa:
                    suma_ocen += 1

    ocena_podobieństwa_artykułu = suma_ocen / len(tagi_artykułu)
    return ocena_podobieństwa_artykułu


def ocena_koncowa_artykułu(ocena_podobieństwa, data_utworzenia, ocena_artykułu, liczba_ocen, X, Y, Z):
    now = datetime.now()
    wiek_artykułu = (now - data_utworzenia).days

    if wiek_artykułu < 7:
        ocena_podobieństwa += X
    elif wiek_artykułu < 14:
        ocena_podobieństwa += 0.75 * X
    elif wiek_artykułu < 21:
        ocena_podobieństwa += 0.5 * X
    elif wiek_artykułu < 28:
        ocena_podobieństwa += 0.25 * X

    if liczba_ocen > Y:
        if ocena_artykułu < 2:
            ocena_podobieństwa -= 0.5 * Z
        elif ocena_artykułu > 4:
            ocena_podobieństwa += Z
        else:
            ocena_podobieństwa += 0.5 * Z

    ocena_końcowa_artykułu = ocena_podobieństwa
    return ocena_końcowa_artykułu
