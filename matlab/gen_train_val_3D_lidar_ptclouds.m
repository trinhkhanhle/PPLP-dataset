% A program to generate 3D synthethic LIDAR pointcloud from 1 specified Kinect and 
% Kinect images into lidar_ptclouds_3D and rgb_images directory.
% 
% Note: the output pointcloud is saved as ply file w.r.t Kinect camera coordinate system.
% The camera image is saved as jpg file.
%
% Note2: 
% Set kidx variable to choose a different Kinect. Default Kinect is 50_01 (kidx=1). 
% Set bVisOutput = 1 to visualize what is going on. Set bVisFolderGen = 1 to save visualization images. Default are 0s
%
% Note3:
% This code assumes that you have already downloaded sequences from CMU Panoptic Dataset website:
% http://domedb.perception.cs.cmu.edu/dataset.html
%
% Note4: This code assumes that your matlab can be run from CLI and has Computer Vision toolbox to
% save and (optionally) visualize point cloud data
% (https://www.mathworks.com/help/vision/3-d-point-cloud-processing.html)
%
% Trinh Le (trle@umich.edu) and Fan Bu (fanbu@umich.edu)

% Input Path Setting  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The following folder hiearchy is assumed:
% (dataset_path)/(seqName)/kinect_shared_depth
% (dataset_path)/(seqName)/kinectImgs
% (dataset_path)/(seqName)/kcalibration_(seqName).json
% (dataset_path)/(seqName)/ksynctables_(seqName).json
% (dataset_path)/(seqName)/calibration_(seqName).json
% (dataset_path)/(seqName)/synctables_(seqName).json

% Output Path Setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (output_path)/(seqName)/lidar_ptclouds_3D
% (output_path)/(seqName)/rgb_images

% Parameters:
% dataset_path: path to CMU dataset parent folder
% output_path: path to where output are stored
% seqName: sequence name, i.e. 160422_ultimatum1
% seqUid: integer unique sequence number that will be part of output filename. Must be unique from 0 to 99 for each sequence.
% seqBegin: video frame number you want to start (see /(seqName)kinectImgs directory to get an idea about frame range)
% seqEnd: video frame number you want to end (see /(seqName)/kinectImgs directory to get an idea about frame range)

% To execute the program, run from command line (no GUI), for ex:
% sudo matlab -nodesktop -nosplash -r "gen_train_val('/dataset_path', '/output_path', '160422_ultimatum1', 0, 1, 29728)"


function gen_train_val_3D_lidar_ptclouds(dataset_path, output_path, seqName, seqUid, seqBegin, seqEnd) 

kidx = 1
kcolor_index_list= seqBegin:seqEnd;
sampling = 7

%Relative Paths
kinectImgDir = sprintf('%s/%s/kinectImgs',dataset_path,seqName);  
kinectDepthDir = sprintf('%s/%s/kinect_shared_depth',dataset_path,seqName);
calibFileName = sprintf('%s/%s/kcalibration_%s.json',dataset_path,seqName,seqName);
syncTableFileName = sprintf('%s/%s/ksynctables_%s.json',dataset_path,seqName,seqName);
panopcalibFileName = sprintf('%s/%s/calibration_%s.json',dataset_path,seqName,seqName);

% Output folder Path

%Change the following if you want to save outputs on another folder
seq_path = sprintf('%s/%s',output_path,seqName);
plyOutputDir=sprintf('%s/%s/training/lidar_ptclouds_3D',output_path,seqName); % was root_path
rgbImageDir=sprintf('%s/%s/training/rgb_images',output_path,seqName);

mkdir(plyOutputDir);
mkdir(rgbImageDir);
disp(sprintf('PLY files will be saved in: %s\',plyOutputDir));

%Other parameters
bVisOutput = 0; %Turn on, if you want to visualize what is going on
bVisFolderGen = 0;
bRemoveCeiling= 1;  %Turn on, if you want to remove points from floor


addpath('jsonlab');
addpath('kinoptic-tools');

%% Load syncTables
ksync = loadjson(syncTableFileName);
knames = {};
for id=1:10; knames{id} = sprintf('KINECTNODE%d', id); end


%% Load Kinect Calibration File
kinect_calibration = loadjson(calibFileName);

panoptic_calibration = loadjson(panopcalibFileName);
% [01_01, 01_02, 01_03, ..., 50_03, 50_04, ...] : HD, VGA, Kinect camera name
panoptic_camNames = cellfun( @(X) X.name, panoptic_calibration.cameras, 'uni', false ); %To search the targetCam

        
for cindex = kcolor_index_list
    
   
    %% Main Iteration    

    for idk = kidx  %Iterating 10 kinects. Change this if you want a subpart
        close all;
        if idk==1 && bVisOutput   %Visualize the results from the frist kinect only. 
            vis_output = 1;
        else
            vis_output = 0;
        end
        
        %% Compute Universal time
        selUnivTime = ksync.kinect.color.(knames{idk}).univ_time(cindex); % cindex 1-based: Matlab is 1-based
        fprintf('cindex: %d, UnivTime: %.3f\n', cindex, selUnivTime)
        
               
        
        %% Select corresponding frame index rgb and depth by selUnivTime
        % Note that kinects are not perfectly synchronized (it's not possible),
        % and we need to consider offset from the selcUnivTime
        [time_dist, dindex] = min( abs(selUnivTime - (ksync.kinect.depth.(knames{idk}).univ_time - 6.25) ) ); %dindex: 1 based


        % Filtering if current kinect data is far from the selected time
        fprintf('idk: %d, %.4f\n', idk, selUnivTime - ksync.kinect.depth.(knames{idk}).univ_time(dindex));
        if abs(ksync.kinect.depth.(knames{idk}).univ_time(dindex) - ksync.kinect.color.(knames{idk}).univ_time(cindex))>6.5
            fprintf('Skipping %d, depth-color diff %.3f\n',  abs(ksync.kinect.depth.(knames{idk}).univ_time(dindex) - ksync.kinect.color.(knames{idk}).univ_time(cindex)));    
            continue;
        end
        
        
        % Extract image and depth
        % RGB image for selected kinect for current frame
        rgbFileName = sprintf('%s/50_%02d/50_%02d_%08d.jpg',kinectImgDir,idk,idk,cindex);
        
        % Depth image for selected kinect
        depthFileName = sprintf('%s/KINECTNODE%d/depthdata.dat',kinectDepthDir,idk);
        try
            rgbim = imread(rgbFileName); % cindex: 1 based
        catch me
            fprintf('Kinect image not found: %08d\n', cindex);
            continue;
        end
        
        depthim = readDepthIndex_1basedIdx(depthFileName,dindex);  % dindex: 1 based
        
        %Check valid pixels
        validMask = depthim~= 0; %Check non-valid depth pixels (which have 0)
        nonValidPixIdx = find(validMask(:)==0);
        %validPixIdx = find(validMask(:)==1);

        if vis_output
            figure; imshow(rgbim);     title('RGB Image');
            figure; imagesc(depthim);  title('Depth Image');
            figure; imshow(validMask*255); title('Validity Mask');
            waitforbuttonpress;
        end

        %% Back project depth to 3D points (in camera coordinate)
        camCalibData = kinect_calibration.sensors{idk};

        % point3d (N x 3): 3D point cloud from depth map in the depth camera cooridnate
        % point2d_color (N x 2): 2D points projected on the rgb image space
        % Where N is the number of pixels of depth image (424 x 512)  
        [point3d, point2d_incolor] = unprojectDepth_release(depthim, camCalibData, true);

        % point3d = (424 x 512) x 3 => that's maximum of pointcloud per image
        % point2d_incolor = (424 x 512) x 2. Special case: < (424 x 512) due to point3d outside image view
        
        validMask = validMask(:) &  ~(point3d(:,1)==0);
        nonValidPixIdx = find(validMask(:)==0);
        point3d(nonValidPixIdx,:) = nan;
        point2d_incolor(nonValidPixIdx,:) = nan;
        
        %% Convert point3d from depth frame to camera frame by using:
        rgbcam.R = camCalibData.M_color(1:3,1:3);
        rgbcam.t = camCalibData.M_color(1:3,4);
        rgbcam.K = camCalibData.K_color;
        point3d_incam = bsxfun(@plus, rgbcam.R * point3d', rgbcam.t)';
        
        point3d_incam(nonValidPixIdx,:) = nan;
        
        % rgbim = (1080 x 1920) x 3
       
        %% Project 3D points (from depth) to color image
        colors_inDepth = multiChannelInterp( double(rgbim)/255, ...
            point2d_incolor(:,1)+1, point2d_incolor(:,2)+1, 'linear'); % colors_inDepth: contains pixels X, Y and their RGB colors; rgbim: 3 channels
        % colors_inDepth = (424 x 512) x 3
        
        colors_inDepth = reshape(colors_inDepth, [size(depthim,1), size(depthim,2), 3]); % (424 x 512)
        colorsv = reshape(colors_inDepth, [], 3); % Nx3: RGB: (424 x 512) x 3
        
        
        % valid_mask = depthim~=0;
        validMask = validMask(:) & ~isnan(point2d_incolor(:,1)); % Check X != NaN
        validMask = validMask(:) & ~isnan(colorsv(:,1)); % Check X != NaN
        %nonValidPixIdx = find(validMask(:)==0);
        validPixIdx = find(validMask(:)==1);

        
        colorsv = colorsv(validPixIdx,:);
        point2d_incolor = point2d_incolor(validPixIdx, :);
        point3d_incam = point3d_incam(validPixIdx,:); % Note that validPixIdx is maintained through various frames
        
        %% Select only pointcloud visible to Kinect RGB
        pt2_x = point2d_incolor(:,1);
        pt2_y = point2d_incolor(:,2);
        idx = find(pt2_x<0 |  pt2_y<0 | pt2_x>1920 | pt2_y>1080 );
        
        colorsv(idx, :) = [];
        point2d_incolor(idx, :) = [];
        point3d_incam(idx, :) = [];
        
        
        %% Filtering point cloud based on height to simulate LIDAR signal
        panoptic_calibData = panoptic_calibration.cameras{find(strcmp(panoptic_camNames, sprintf('50_%02d', idk)))};
        M = [panoptic_calibData.R, panoptic_calibData.t];
        T_panopticWorld2KinectColor = [M; [0 0 0 1]]; %Panoptic_world to Kinect_color
        T_kinectColor2PanopticWorld = inv(T_panopticWorld2KinectColor);
        
        scale_kinoptic2panoptic = eye(4);
        scaleFactor = 100;%0.01; %centimeter to meter
        scale_kinoptic2panoptic(1:3,1:3) = scaleFactor*scale_kinoptic2panoptic(1:3,1:3);
        
               
        point3d_pan = T_kinectColor2PanopticWorld* scale_kinoptic2panoptic* [point3d_incam'; ones(1, size(point3d_incam,1))];
        point3d_pan = point3d_pan(1:3,:)';
        point3d_pan = double(point3d_pan);
        %% Delete floor light
        if bRemoveCeiling
            % Delete floor points
            % Crop floor 
            floorPtIdx =(find(point3d_pan(:,2)<-300));      %Up-direction => negative Y axis
            point3d_pan(floorPtIdx,:) =[];
            colorsv(floorPtIdx,:) =[];         
        end

        
        % Project back to Kinect color sensor
        point3d_lidar_cam = bsxfun(@plus, panoptic_calibData.R * point3d_pan', panoptic_calibData.t(:))' ./ 100.0;
        
        
        

        
        out_pc_color = pointCloud(point3d_lidar_cam(1:sampling:end,:));
        out_pc_color.Color = uint8(round(colorsv(1:sampling:end,:)*255));
        out_fileName_color = sprintf('%s/%02d%02d%02d%06d.ply', plyOutputDir, 50, idk, seqUid, cindex);
        pcwrite(out_pc_color, out_fileName_color,'PLYFormat','ascii');
        
        
        
        %% Visualization

        % Display depth map
        if vis_output
            %Note that this image has an artifact since no z-buffering are performed 
            %That is, occluded part may extract colors from RGB image
            figure; imshow(colors_inDepth); 
            title('Depth map, after extracting colors from RGB image');
            waitforbuttonpress;
        end
        
        if vis_output
            figure; scatter3(point3d_lidar_cam(1:sampling:end,1),point3d_lidar_cam(1:sampling:end,2),point3d_lidar_cam(1:sampling:end,3),'.'); axis equal;
            view(0,0); % Bird Eye's View
            title('Point Cloud LIDAR in cam frame');
            waitforbuttonpress;
            
        end
        
        if vis_output
            [point2d_lidar_color] = PoseProject2D(point3d_pan(1:sampling:end, :), panoptic_calibData, 1); % This is used for visualization
            figure; scatter(point2d_lidar_color(:,1),point2d_lidar_color(:,2), 1,colorsv(1:sampling:end, :));  axis equal; % Each point cloud is a circle with size = 1
            title('Pt cloud LIDAR in Image Frame');
            view(2);
            waitforbuttonpress;
        end
        
        % Display point cloud projected on Kinect RGB
        if bVisOutput
            
            figure(2); 
            imshow(rgbim);
            hold on;
            
            plot(point2d_lidar_color(:,1),point2d_lidar_color(:,2),'.');
            axis equal;
            
            if bVisFolderGen
                vizOutputDir = sprintf('%s/%s/training/visual_verification',output_path,seqName);
                mkdir(vizOutputDir);
                
                rgb_ptcloud_filename = sprintf('%s/%02d%02d%02d%06d.jpg',vizOutputDir, 50, idk, seqUid, cindex);
                
                set(gcf,'PaperUnits','inches','PaperPosition',[0 0 19.2 10.8])
                print(rgb_ptcloud_filename,'-djpeg','-r100')
            end
            
            
            waitforbuttonpress;       
        end
        
        
    end

end

end

