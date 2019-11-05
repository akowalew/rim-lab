function [yorg, yin, yref, Fs] = audiofir(file)
    %% odczyt oryginalnego sygnału
    [yorg, Fs] = audioread(file); % yorg = [y_left, y_right]
    len = length(yorg);
    assert(Fs == 44100);
    assert(size(yorg, 2) == 2);

    %% zakłócenie sygnału szumem i przydźwiękiem sieciowym
    noise = filter(fir1(128, 11000/(Fs/2), 'high',...
                        blackmanharris(129)), 1,...
                        randn(size(yorg))); % szum w.cz.

    hum = sin(2*pi*50*(0:len-1).'/Fs); % przydźwięk 50Hz
    yin = yorg + noise + [hum, hum];

    %% zaprojektowanie filtru i odfiltrowanie zakłóceń
    n = 1024;
    coeff = firpm(n, [0 70 90 9900 10100 (Fs/2)]/(Fs/2), [0 0 1 1 0 0 ],[10.1606 1 10000]);

    tic
    yref = filter(coeff, 1, yin);
    toc

    %% zapamiętanie filtru i sygnałów w pliku binarnym
    f = fopen('audiofir_in.dat', 'wb');
    fwrite(f, [n, len], 'int');
    fwrite(f, coeff, 'single');
    fwrite(f, yin,
    'single');
    fwrite(f, yref, 'single');
    fclose(f);
