%% Odtwarzanie sygnału kompresywnie spróbkowanego metodą
%% kroczącego dopasowania (Matching Pursuit)
    %% Definicja rozmiarów problemu
    N = 365; % długość oryginalnego sygnału
    K = 20; % liczba niezerowych próbek sygnału
    M = 50; % liczba obserwacji przetworzonego sygnału

    %% Generacja oryginalnego sygnału do odtworzenia
    x = zeros(N, 1); % rezerwacja miejsca na oryginalny sygnał
    ix = randperm(N); % losowy wybór indeksów próbek sygnału
    x(ix(1:K)) = abs(randn(K, 1)); % wstawienie losowych wartości

    %% Generacja losowej macierzy pomiarowej
    tau = 30; % stała czasowa zaniku sygnału obserwowanego
    Ap = toeplitz(exp(-(0:N-1)/tau), [1, zeros(1, N-1)]);
    iA = randperm(N); % losowy wybór indeksów pomiarów
    A = Ap(iA(1:M), :); % macierz (losowo wybranych) pomiarów
    y = A * x; % wektor obserwacji

    %% Algorytm kroczącego dopasowania (Matching Pursuit)
    xr = zeros(N, 1); % zerowanie miejsca na odtworzony sygnał
    r = y; % początkowo "pozostałością" jest sygnał obserwowany
    nrm2y = norm(y);
    nrm2r = nrm2y;

    t = 1; % licznik iteracji alorytmu
    while nrm2r > 0.05 * nrm2y && t <= 50 % główna pętla algorytmu
        sp = A.' * r; % obliczenie wektora iloczynów skalarnych
        [dummy, i] = max(abs(sp)); % znalezienie maks. z nich,
        nrm2a = norm(A(:, i)); % norma odpowiedniej kolumny A
        s = sp(i); % pobranie maksymalnego iloczynu skalarnego
        s = s / nrm2a^2; % unormowanie udziału w "pozostałości"
        xr(i) = xr(i) + s; % wstawienie udziału do odtwarzanego x
        r = r - s * A(:, i); % aktualizacja wektora "pozostałości"
        nrm2r = norm(r); % obliczenie normy "pozostałości"
        fprintf('iter.%3d: x(%3d) <- %4.2f, nrm2res=%4.2f\n', t, i, s, nrm2r); % wydruk kontrolny
        t = t + 1; % aktualizacja licznika iteracji
    end

    %% Graficzna prezentacja wyników działania algorytmu
    Apx = Ap * x; % wszystkie próbki sygnału obserwowanego
    subplot(311); plot(x);
        ylabel('sygn. oryg.');
    subplot(312); plot(1:N,Apx, iA(1:M),Apx(iA(1:M)),'g*');
        ylabel('sygn. obs.');
    subplot(313); plot(xr);
        ylabel('sygn. odtw.'); xlabel('czas');
    title('Odtwarzanie sygnalu metoda kroczacego dopasowania');

    %% Zapamiętanie w pliku binarnym danych i wyników
    f = fopen('raindemo.dat', 'wb');
    fwrite(f, [M, N], 'int');
    fwrite(f, x, 'single');
    fwrite(f, A, 'single');
