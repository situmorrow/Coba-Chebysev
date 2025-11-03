function draw_collision_free(array_state,array_state_history, ...
    reference_pose,total_time_index,axis_lim,Np,situation,time)

% --- guard ukuran data ---
assert(size(array_state,1)==5, 'array_state harus 5 x T');
assert(numel(time)==size(array_state,2), 'Panjang time harus = kolom array_state');
assert(size(array_state_history,1)==Np+2 && size(array_state_history,2)==5, ...
    'array_state_history harus (Np+2) x 5 x K');

set(0,'DefaultAxesFontName','Times New Roman');
set(0,'DefaultAxesFontSize',12);

line_width = 0.8;
fontsize_labels = 14;
mid_progress = round(1+total_time_index/2);

% --- parameter gambar kapal ---
shipLength = 101.7; shipWidth = 40;
ship_polygon = [ ...
    [shipLength/2, shipLength/6, -shipLength/2, -shipLength/2, shipLength/6, shipLength/2]
    [0,            -shipWidth/2, -shipWidth/2,  shipWidth/2,   shipWidth/2,  0] ];

% --- lingkaran sekitar kapal (opsional) ---
circle = 0:0.02:2*pi;
x_circle = (shipLength/2)*cos(circle);
y_circle = (shipLength/2)*sin(circle);

% --- posisi referensi ---
x_ref = reference_pose(1); 
y_ref = reference_pose(2); 
heading_ref = reference_pose(3);

% --- siapkan figure & video ---
video_name = strcat(situation,'_video','.mp4');
fig = figure(500); clf(fig);
set(gcf,'PaperPositionMode','auto','Color','w','Units','normalized','OuterPosition',[0 0 0.55 1]);

vw = VideoWriter(video_name,'MPEG-4'); vw.FrameRate = 15; open(vw);

% --- gambar elemen statis sekali ---
axes('Parent',gcf); hold on; box on; grid on;
axis(axis_lim); axis equal;
xlabel('$x_E$-position (m)','interpreter','latex','FontSize',fontsize_labels);
ylabel('$y_E$-position (m)','interpreter','latex','FontSize',fontsize_labels);
title(['Ship Trajectory (' situation ')']);

% tujuan (ref ship)
Rref = [cos(heading_ref) -sin(heading_ref); sin(heading_ref) cos(heading_ref)];
ref_poly = Rref*ship_polygon + [x_ref; y_ref];
hRef = fill(ref_poly(1,:), ref_poly(2,:), [0.2 0.2 0.2], 'EdgeColor','none', 'FaceAlpha',0.25); %#ok<NASGU>
plot(x_ref, y_ref, 'rx', 'MarkerSize', 10, 'LineWidth', 2);

% handle dinamis
h_path = plot(nan, nan, 'k-', 'LineWidth', 1.2);
h_ship = fill(nan, nan, 'y', 'EdgeColor','k', 'LineWidth', 0.5);
h_pred = plot(nan, nan, 'g--', 'LineWidth', 1.0);
h_ring = plot(nan, nan, '--k', 'LineWidth', 0.5);
h_time = text(axis_lim(1)+20, axis_lim(3)+40, '', 'FontSize', 12, 'Color', 'k');

x_hist = []; y_hist = [];

T = size(array_state,2);
K = size(array_state_history,3);

for k = 1:T
    % --- data frame ke-k ---
    x_os = array_state(3,k); 
    y_os = array_state(4,k); 
    heading_os = array_state(5,k);
    x_hist = [x_hist x_os]; %#ok<AGROW>
    y_hist = [y_hist y_os];

    % --- kapal OS ---
    Ros = [cos(heading_os) -sin(heading_os); sin(heading_os) cos(heading_os)];
    os_poly = Ros*ship_polygon + [x_os; y_os];

    % --- prediksi horizon ---
    if k <= K
        hist_k = array_state_history(:,:,k);
    else
        hist_k = array_state_history(:,:,end);
    end
    x_pred = hist_k(2:Np+1,3);
    y_pred = hist_k(2:Np+1,4);

    % --- update grafik (tanpa clf supaya video halus) ---
    set(h_path,'XData',x_hist,'YData',y_hist);
    set(h_ship,'XData',os_poly(1,:),'YData',os_poly(2,:));
    set(h_pred,'XData',x_pred,'YData',y_pred);
    set(h_ring,'XData',x_os + x_circle, 'YData', y_os + y_circle);
    set(h_time,'String',sprintf('t = %.1f s', time(k)));

    drawnow limitrate;

    % --- rekam frame ---
    F = getframe(gcf);
    writeVideo(vw, F);
end

% simpan gambar ringkas
saveas(gcf, [situation '_trajectory.png']);
close(vw);

end
