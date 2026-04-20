% =========================================================================
% Circadian Photometry Analysis Pipeline
% Calculates rolling least-squares dF/F and extracts Area Under the Curve (AUC)
% =========================================================================

% dbstop if naninf
FileList = dir('*.csv');
N = numel(FileList); 

% Pre-allocate cell arrays
data_cell = cell(1, N);
signal_raw = cell(1,N);
signal_pass = cell(1,N);
uv_pass = cell(1,N);
uv_raw = cell(1,N);
dff = cell(1,N);
new_uv_plot = cell(1,N);
baseline = cell(1,N);

% =========================================================================
% --- 1. Data Pre-processing ---
% =========================================================================
fc = 1;              % Low-pass filter cutoff (Hz)
fs = 20;             % Sampling frequency (Hz)
Wn = fc / (fs / 2);  % Normalize cutoff frequency
[b_filt, a_filt] = butter(3, Wn);  

window = 3000;       % Moving average window for detrending

disp('--- Loading and Filtering Raw Data ---');
for i = 1:N
    fileName = FileList(i).name;
    data = readtable(fileName, 'Headerlines', 2);
    Data_array = table2array(data);
    
    LED = Data_array(:,3);
    signal_rows = (LED == 2);
    uv_rows = (LED == 1);
    column_range = 9:16;
    
    signal_data = Data_array(signal_rows, column_range);
    uv_data = Data_array(uv_rows, column_range);
    
    % Trim edges to remove artifactual spikes
    signal_data = signal_data(100:end-40,:);
    uv_data = uv_data(100:end-40,:);
    
    min_rows = min(size(signal_data, 1), size(uv_data, 1));
    signal_data = signal_data(1:min_rows, :);
    uv_data = uv_data(1:min_rows, :);

    signal_raw{i} = signal_data + 1;
    uv_raw{i} = uv_data + 1;
end

for i = 1:N
    signal_data = signal_raw{i};
    uv_data = uv_raw{i};
    
    signal_pass{i} = zeros(size(signal_data));
    uv_pass{i} = zeros(size(uv_data));

    % Detrend and apply low-pass filter
    for r = 1:8
        signal_base = movmean(signal_data(:, r), window);
        uv_base = movmean(uv_data(:, r), window);
        
        signal_detrend = signal_data(:, r) - signal_base;
        uv_detrend = uv_data(:, r) - uv_base;
        
        signal_filt = filtfilt(b_filt, a_filt, signal_detrend);
        uv_filt = filtfilt(b_filt, a_filt, uv_detrend);
        
        signal_pass{i}(:, r) = signal_filt + 1;
        uv_pass{i}(:, r) = uv_filt + 1;
    end
end

% =========================================================================
% --- 2. Rolling Least-Squares Regression & dF/F Calculation ---
% =========================================================================
disp('--- Starting Analysis: Rolling Least-Squares Regression ---');

window_size_pts = 3000; 

for i = 1:N
    current_signal_data = signal_pass{i};
    current_uv_data = uv_pass{i};
    denominator_data = uv_raw{i};
    
    num_points = size(current_signal_data, 1);
    
    dff{i} = zeros(num_points, 8);
    new_uv_plot{i} = zeros(num_points, 8);
    baseline{i} = zeros(num_points, 8); 

    for r = 1:8
        signal_col = current_signal_data(:, r);
        uv_col = current_uv_data(:, r);
        
        % Calculate Rolling Statistics 
        rolling_mean_signal = movmean(signal_col, window_size_pts, 'omitnan');
        rolling_mean_uv = movmean(uv_col, window_size_pts, 'omitnan');
        rolling_var_uv = movvar(uv_col, window_size_pts, 'omitnan');
        
        % Calculate rolling covariance
        rolling_cov = movmean((signal_col - rolling_mean_signal) .* (uv_col - rolling_mean_uv), window_size_pts, 'omitnan');

        % Calculate Rolling Regression Coefficients (a*x + b)
        rolling_a = rolling_cov ./ rolling_var_uv; % Slope
        rolling_b = rolling_mean_signal - (rolling_a .* rolling_mean_uv); % Intercept
        
        % Handle NaNs at edges 
        rolling_a = fillmissing(rolling_a, 'previous', 'EndValues', 'nearest');
        rolling_a = fillmissing(rolling_a, 'next', 'EndValues', 'nearest');
        rolling_b = fillmissing(rolling_b, 'previous', 'EndValues', 'nearest');
        rolling_b = fillmissing(rolling_b, 'next', 'EndValues', 'nearest');
        
        % Calculate fitted UV trace
        fitted_uv = (rolling_a .* uv_col) + rolling_b;
        new_uv_plot{i}(:, r) = fitted_uv;
        
        % Calculate dF/F using raw UV data as denominator
        denominator_col = denominator_data(:, r);
        denominator_col(denominator_col == 0) = 1; % Prevent division by zero
        
        dff_col = ((signal_col - fitted_uv) ./ denominator_col) * 100;
        
        % Set negative dF/F to zero
        dff_col(dff_col < 0) = 0;
        dff{i}(:, r) = dff_col;

        % Store a moving median of the dF/F trace as dynamic baseline
        baseline{i}(:, r) = movmedian(dff_col, window_size_pts, 'omitnan');
    end
    
    if mod(i, 10) == 0 || i == N
        fprintf('Processed file %d of %d\n', i, N);
    end
end
disp('--- Rolling Regression Analysis Complete ---');

% =========================================================================
% --- 3. Post-Processing: Final Filter & Area Under Curve (AUC) ---
% =========================================================================

% Apply final low-pass Butterworth filter to dF/F data
for r = 1:8
    for i = 1:N
        if ~isempty(dff{i})
            dff{i}(:, r) = filtfilt(b_filt, a_filt, dff{i}(:, r));
        end
    end
end
disp('--- Final dF/F filtering complete ---');

% Calculate AUC (Area Under the Curve)
disp('--- Calculating Area Under the Curve (AUC) ---');
area_jor = zeros(N, 8);

for i = 1:N
    dff_above_baseline = dff{i} - baseline{i}; 
    dff_above_baseline(dff_above_baseline < 0) = 0; 
    area_jor(i, :) = trapz(dff_above_baseline);
end

% Baseline correct the AUC measurements
area_jor_corr = zeros(N, 8);
for j = 1:8
    area_jor_corr(:,j) = area_jor(:,j) - movmean(area_jor(:,j), 100);
end
area_jor_corr = area_jor_corr + 50; % Offset for downstream alignment

% =========================================================================
% --- 4. Graphical Representation (Plot Looping) ---
% Note: Requires 'tight_subplot.m' on MATLAB path
% =========================================================================
disp('--- Generating Summary Plots ---');

dawn_orient = 0; % Configurable offset for start of day
ch = 3;          % Selected channel for plotting

% Generate baseline-adjusted UV plot for visual comparison
uv_raw_test = cell(size(uv_raw)); 
for i = 1:length(uv_raw)
    uv_raw_test{i} = uv_pass{i} + (median(signal_pass{i}) - max(uv_pass{i}));
end

% Set up custom colors
firehouse = [178 34 34] / 255;
blues = '#4F86F7';

% Loop through specific 48-hour segments (adjust range as needed)
for k = 5:5  
    
    % Map indices for Day and Night cycles
    day = zeros(1, 12);
    night = zeros(1, 12);
    for s = 1:12
        day(s) = (s*2) + dawn_orient + (48 * k);
        night(s) = (s*2) + 24 + dawn_orient + (48 * k);
    end

    % --------- Figure 1: Day DFF ---------
    max_marg_h = max(0.05, 0.1);
    marg_v = 0.1;
    total_height = 12;
    subplot_height = (total_height - marg_v * 62) / 12;
    
    figure('Units', 'inches', 'Position', [0, 0, 4, 8], 'Color', 'white');
    y_dff = [-0.01, 0.9];
    ha = tight_subplot(12, 1, [0.02, 0.05], [0.1, marg_v], [max_marg_h, 0.05]);

    for j = 1:numel(day)
        axes(ha(j));
        plot(dff{day(j)}(:,ch), 'Color', firehouse, 'LineWidth', 1);
        box off; axis off; ylim(y_dff);
        set(gca, 'Color', 'white');
        ha(j).Position(4) = subplot_height;
    end

    % --------- Figure 2: Night DFF ---------
    figure('Units', 'inches', 'Position', [0, 0, 4, 8], 'Color', 'white');
    ha = tight_subplot(12, 1, [0.02, 0.05], [0.1, marg_v], [max_marg_h, 0.05]);

    for j = 1:numel(night)
        axes(ha(j));
        plot(dff{night(j)}(:,ch), 'Color', firehouse, 'LineWidth', 1);
        axis off; ylim(y_dff);
        set(gca, 'Color', 'white');
        ha(j).Position(4) = subplot_height;
    end

    % --------- Figure 3: Day Raw Signals (Signal vs Control) ---------
    marg_h = 0.0;
    marg_v_top = -0.05;
    marg_v_bottom = 0.1;
    y_raw = [0.96, 1.1];

    figure('Units', 'inches', 'Position', [0, 0, 4, 8], 'Color', 'white');
    ha = tight_subplot(12, 1, [0.01, 0.05], [marg_v_top, marg_v_bottom], [marg_h, marg_h]);

    for j = 1:numel(day)
        axes(ha(j));
        plot(signal_pass{day(j)}(:,ch), '-m', 'LineWidth', 1); hold on;
        plot(uv_pass{day(j)}(:,ch), 'Color', blues, 'LineWidth', 1);
        box off; axis off; ylim(y_raw);
        set(gca, 'Color', 'white');
        ha(j).Position(4) = subplot_height;
    end

    % --------- Figure 4: Night Raw Signals (Signal vs Control) ---------
    figure('Units', 'inches', 'Position', [0, 0, 4, 8], 'Color', 'white');
    ha = tight_subplot(12, 1, [0.01, 0.05], [marg_v_top, marg_v_bottom], [marg_h, marg_h]);
    
    for j = 1:numel(night)
        axes(ha(j));
        plot(signal_pass{night(j)}(:,ch), '-m', 'LineWidth', 1); hold on;
        plot(uv_pass{night(j)}(:,ch), 'Color', blues, 'LineWidth', 1);
        box off; axis off; ylim(y_raw);
        set(gca, 'Color', 'white');
        ha(j).Position(4) = subplot_height;
    end
end
disp('--- Script Complete ---');