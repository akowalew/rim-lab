function [yorg, yin, yref, Fs] = testfir
    %% generacja oryginalnego sygnału
    Fs = 44100;
    yorg = [1 2 3 4 5 6 7 8 9];
    yorg = [yorg.', -yorg.'];
    len = length(yorg);

    %% brak zakłócenia sygnału szumem i przydźwiękiem sieciowym
    yin = yorg;

    %% zaprojektowanie filtru i filtracja
    coeff = [1e0 1e1 1e2 1e3];
    n = length(coeff) - 1;
    yref = filter(coeff, 1, yin);

    %% zapamiętanie filtru i sygnałów w pliku binarnym
    f = fopen('audiofir_in.dat', 'wb');
    fwrite(f, [n, len], 'int');
    fwrite(f, coeff, 'single');
    fwrite(f, yin, 'single');
    fwrite(f, yref, 'single');
    fclose(f);
