%% CameraSimulation
% Simulation of a monocular camera without any distortion
% Possibility to compute the vanishing point by setting the adequat boolean to 1
% The program project a XY plan into the camera 2D image. 
% You can adjust the Euler angles with the 3 sliders to change the camera orientation.
% To modify the camera position you need to do it manually by modifying the variable O_Rcam.
% Contact: pierremarie.damon@gmail.com

 
function CameraSimulation

    bool_vanishing_point=0; % set the vanishing point computation option here
    
    f = figure('units','normalized','outerposition',[0 0 1 1]); 
    
    % Camera definition
    % Intrinsic matrix, camera_parameters=[fx 0 x0; 0 fy y0; 0 0 1]
    % fx, fy:  focal lengths 
    % x0, y0: optical center projection coordinates in the 2D cxamera space
    fx = 1000; fy = 1000; 
    x0 = 500; y0 = 500; 
    camera_parameters = [fx  0   x0 ; 
                         0   fy  y0 ; 
                         0   0   1  ];

    % Image resolution, image_size=[nb_horizontal_pixels, nb_vertical_pixels]
    image_size = [1000,1000];

    % 3D camera orientation  
    Rot_cam = [ 0    0    1;
               -1    0    0;
                0   -1    0]';
            
    yaw     =  0*pi/180; % Initial orientation
    roll    =  0*pi/180; % Initial orientation
    pitch   =  0*pi/180; % Initial orientation

    % 3D camera position
    O_Rcam = [-20;0;5]; % Origine of the camera frame coordinate

    % Projection & transformation matrices
    rotation_matrix      = rotationVectorToMatrix([pitch,yaw,roll])*Rot_cam;
    translation_matrix   = O_Rcam;
    projection_camera    = [rotation_matrix -rotation_matrix*translation_matrix];
    
    % Environment definition
    % XY Plan
    x_min = 0;  x_max = 50; dx = 25;
    y_min = -5; y_max = 5;  dy = 5;
    [x_plan,y_plan] = meshgrid([x_min:dx:x_max],[y_min:dy:y_max]); 
    z_plan = 0*x_plan + 0*y_plan;
    
    % Corners identification
    i_corner_1 = 1; 
    i_corner_2 = size(x_plan,1)*(size(x_plan,2)-1)+1;
    i_corner_3 = size(x_plan,1)*size(x_plan,2);
    i_corner_4 = size(x_plan,1);
   
    % Give a specific color to each 3D points
    points_3D = [x_plan(:),y_plan(:),z_plan(:)];
    color     = hsv(length(x_plan(:)));
    
    % Projection into the camera coordinate system
    points_3D_Rcam = ([projection_camera;  0 0 0 1]*[points_3D,ones(size(points_3D,1),1)]')';

    % Transformation of the XY plan into the 2D image
    points_2D = ([camera_parameters [0;0;0]]*points_3D_Rcam')'; 
    points_2D = points_2D(:,1:2)./repmat(points_2D(:,3),1,2); % Normalization
    points_2D = [points_2D,color];
    
    % Corners positions into the 2D image
    c1 = points_2D(i_corner_1,:);
    c2 = points_2D(i_corner_2,:);
    c3 = points_2D(i_corner_3,:);
    c4 = points_2D(i_corner_4,:);
    
    if (bool_vanishing_point)
        % Computation of the vanishing point
        x = [0:image_size(1):image_size(1)]; % Vector to compute the vanishing point

        % Vanishing line 1:
        a1 = (c2(2)-c1(2))/(c2(1)-c1(1));
        b1 = c1(2)-a1*c1(1);
        y1 = a1*x+b1;

        % Vanishing line 2:
        a2 = (c4(2)-c3(2))/(c4(1)-c3(1));
        b2 = c3(2)-a2*c3(1);
        y2 = a2*x+b2;

        % Intersection of the vanishing lines
        vanishingPoint = InterX([x;y1],[x;y2]);
    end
    
    %% Plot 3D scenario
    f1 = subplot(1,2,1);hold on; grid on; view(50, 30); axis equal; axis([-25 55 -10 10 0 20]);
    xlabel('X');ylabel('Y');zlabel('Z'); title('3D scene');
    
    % Plot XY plan
    surf(x_plan,y_plan,z_plan,'FaceColor','c');
    plan_XY = scatter3(points_3D(:,1),points_3D(:,2),points_3D(:,3),50,color,'filled','MarkerEdgeColor','k');

    % Plot camera
    cam = plotCamera('Location',translation_matrix,'Orientation',rotation_matrix,'Size',1,'AxesVisible',0);

    hold off;   
    
    %% Plot 2D scenario
    f2 = subplot(1,2,2); hold on; grid off; axis off; axis equal;set(gca,'YDir','Reverse'),title('2D image');

    % plot the projected XY surface
    patche = patch([c1(1) c2(1) c3(1) c4(1)],[c1(2) c2(2) c3(2) c4(2)],'c');
    plan_image=scatter(points_2D(:,1),points_2D(:,2),50,[points_2D(:,3) points_2D(:,4) points_2D(:,5)],'filled','MarkerEdgeColor','k');
    
    if (bool_vanishing_point)
        % Plot the optical center
        plot(x0,y0,'k*');

        % Plot the vaniching lines and point
        vanishing_line_1 = plot(x,y1,'k','linewidth',1);
        vanishing_line_2 = plot(x,y2,'k','linewidth',1);
        if (~isempty(vanishingPoint)) 
            vanishing_point  = plot(vanishingPoint(1),vanishingPoint(2),'+k','MarkerSize',10);
        else
            vanishing_point  = plot(x0,y0,'+k','MarkerSize',10,'visible','off');
        end
    end
    
    
    % Plot the image Border
    xlim([0 image_size(1)]); ylim([0 image_size(2)]);
    plot([0 image_size(1)],[0 0],'k','linewidth',3); 
    plot([image_size(1) image_size(1)],[0 image_size(2)],'k','linewidth',3); 
    plot([0 image_size(1)],[image_size(2) image_size(2)],'k','linewidth',3);
    plot([0 0],[0 image_size(2)],'k','linewidth',3);
    
    %% Display the user intarface boxes   
    % Slide bar for pitch value
    label_box_pitch = uicontrol('Style','text','string','Pitch (degree)','Position',[130,40,130,40]);
    slider_pitch = javax.swing.JSlider;
    javacomponent(slider_pitch,[00,20,400,45]);
    set(slider_pitch, 'Value',0, 'MajorTickSpacing',5, 'PaintLabels',true,'PaintTicks',true,'minimum',-30,'maximum',30);
    hslider_pitch = handle(slider_pitch, 'CallbackProperties');
    set(hslider_pitch, 'StateChangedCallback', @update);
    
    % Slide bar for roll value
    label_box_roll = uicontrol('Style','text','string','Roll (degree)','Position',[470,40,500,40]);
    slider_roll = javax.swing.JSlider;
    javacomponent(slider_roll,[520,20,400,45]);
    set(slider_roll, 'Value',0, 'MajorTickSpacing',5, 'PaintLabels',true,'PaintTicks',true,'minimum',-30,'maximum',30);
    hslider_roll = handle(slider_roll, 'CallbackProperties');
    set(hslider_roll, 'StateChangedCallback', @update);
    
    % Slide bar for yaw value
    label_box_yaw= uicontrol('Style','text','string','Yaw (degree)','Position',[980,40,500,40]);
    slider_yaw = javax.swing.JSlider;
    javacomponent(slider_yaw,[1040,20,400,45]);
    set(slider_yaw, 'Value',0, 'MajorTickSpacing',5, 'PaintLabels',true,'PaintTicks',true,'minimum',-30,'maximum',30);
    hslider_yaw = handle(slider_yaw, 'CallbackProperties');
    set(hslider_yaw, 'StateChangedCallback', @update);

   
    function update(~,~)
        % Get new values
        pitch  = get(slider_pitch,'Value')*pi/180;
        roll   = get(slider_roll,'Value')*pi/180;
        yaw    = get(slider_yaw,'Value')*pi/180;
        
        % Compute the new camera orientation
        rotation_matrix   = rotationVectorToMatrix([pitch,yaw,roll])*Rot_cam;
        projection_camera = [rotation_matrix -rotation_matrix*translation_matrix];
        cam.Orientation   = rotation_matrix;

        % Projection into the camera coordinate system
        points_3D_Rcam = ([projection_camera;  0 0 0 1]*[points_3D,ones(size(points_3D,1),1)]')';

        % Transformation of the XY plan into the 2D image
        points_2D = ([camera_parameters [0;0;0]]*points_3D_Rcam')'; 
        points_2D = points_2D(:,1:2)./repmat(points_2D(:,3),1,2); % Normalization
        points_2D = [points_2D,color];

        % Detection of the corners
        c1 = points_2D(i_corner_1,:);
        c2 = points_2D(i_corner_2,:);
        c3 = points_2D(i_corner_3,:);
        c4 = points_2D(i_corner_4,:);
        
        if (bool_vanishing_point)
            % Computation of the vanishing point
            x = [0:image_size(1):image_size(1)]; % Vector to compute the vanishing point

            % Vanishing line 1:
            a1 = (c2(2)-c1(2))/(c2(1)-c1(1));
            b1 = c1(2)-a1*c1(1);
            y1 = a1*x+b1;

            % Vanishing line 2:
            a2 = (c4(2)-c3(2))/(c4(1)-c3(1));
            b2 = c3(2)-a2*c3(1);
            y2 = a2*x+b2;

            % Intersection of the vanishing lines
            vanishingPoint = InterX([x;y1],[x;y2]);
        end
        
        %% 2D plot
        set(plan_image,'XData',points_2D(:,1),'YData',points_2D(:,2))
        set(patche,'XData',[c1(1) c2(1) c3(1) c4(1)],'YData',[c1(2) c2(2) c3(2) c4(2)]);
        if (bool_vanishing_point)       
             set(vanishing_line_1,'XData',x,'YData',y1);
             set(vanishing_line_2,'XData',x,'YData',y2);
             if (~isempty(vanishingPoint)) 
                set(vanishing_point,'XData',vanishingPoint(1),'YData',vanishingPoint(2),'visible','on');
             else
                set(vanishing_point,'XData',x0,'YData',y0,'visible','off');
             end
        end
    end

end