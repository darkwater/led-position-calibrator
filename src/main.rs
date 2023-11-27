use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        RwLock,
    },
    thread,
    time::Duration,
};

use eframe::{
    egui::{self, Area, DragValue, Image, TextureOptions, Window},
    epaint::{Color32, ColorImage, Pos2, Rect, Stroke, TextureHandle, Vec2},
};
use opencv::{
    core::{in_range, Mat_AUTO_STEP, Point, Scalar, Vector, CV_8UC3},
    imgproc::{
        bounding_rect, cvt_color, find_contours, moments, CHAIN_APPROX_SIMPLE, COLOR_RGB2HSV,
        RETR_EXTERNAL,
    },
    prelude::*,
};
use video_rs::{Decoder, Locator, Url};

fn main() {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "LED Position Calibrator",
        native_options,
        Box::new(|cc| Box::new(CalibratorApp::new(cc))),
    )
    .unwrap();
}

struct CalibratorApp {
    image: TextureHandle,
}

static IMAGE: RwLock<Vec<u8>> = RwLock::new(Vec::new());
static IMAGE_WIDTH: AtomicUsize = AtomicUsize::new(0);

static POINTS: RwLock<Vec<Rect>> = RwLock::new(Vec::new());

struct Settings {
    lower_h: f64,
    lower_s: f64,
    lower_v: f64,
    upper_h: f64,
    upper_s: f64,
    upper_v: f64,
}
static mut SETTINGS: Settings = Settings {
    lower_h: 40.0,
    lower_s: 100.0,
    lower_v: 100.0,
    upper_h: 70.0,
    upper_s: 255.0,
    upper_v: 255.0,
};

impl CalibratorApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let ctx = &cc.egui_ctx;
        let image = ctx.load_texture("video feed", ColorImage::example(), TextureOptions::LINEAR);

        thread::spawn({
            let mut image = image.clone();
            move || {
                let opts = video_rs::Options::new_with_rtsp_transport_tcp_and_sane_timeouts();
                let mut decoder = Decoder::new_with_options(
                    &Locator::Url(Url::parse("rtsp://192.168.0.101").unwrap()),
                    &opts,
                )
                .expect("Failed to create decoder");

                for frame in decoder.decode_raw_iter() {
                    let frame = frame.expect("Failed to decode frame");

                    *IMAGE.write().unwrap() = frame.data(0).to_vec();
                    IMAGE_WIDTH.store(frame.width() as usize, Ordering::Relaxed);

                    image.set(
                        ColorImage::from_rgb(
                            [frame.width() as usize, frame.height() as usize],
                            frame.data(0),
                        ),
                        TextureOptions::LINEAR,
                    );
                }
            }
        });

        thread::spawn({
            move || {
                loop {
                    thread::sleep(Duration::from_millis(100));

                    let image_data = IMAGE.read().unwrap().clone();

                    let image = {
                        let width = IMAGE_WIDTH.load(Ordering::Relaxed);

                        if width == 0 {
                            continue;
                        }

                        unsafe {
                            Mat::new_rows_cols_with_data(
                                (image_data.len() / width / 3) as i32,
                                width as i32,
                                CV_8UC3,
                                image_data.as_ptr() as *mut _,
                                Mat_AUTO_STEP,
                            )
                            .unwrap()
                        }
                    };

                    let mut hsv_image = Mat::default();
                    cvt_color(&image, &mut hsv_image, COLOR_RGB2HSV, 0).unwrap();
                    drop((image, image_data));

                    let settings = unsafe { &SETTINGS };
                    let lower_green =
                        Scalar::new(settings.lower_h, settings.lower_s, settings.lower_v, 0.0);
                    let upper_green =
                        Scalar::new(settings.upper_h, settings.upper_s, settings.upper_v, 0.0);

                    // Threshold the HSV image to get only green colors
                    let mut mask = Mat::default();
                    in_range(&hsv_image, &lower_green, &upper_green, &mut mask).unwrap();

                    // Find contours
                    let mut contours = Vector::<Vector<Point>>::new();
                    find_contours(
                        &mask,
                        &mut contours,
                        RETR_EXTERNAL,
                        CHAIN_APPROX_SIMPLE,
                        Default::default(),
                    )
                    .unwrap();

                    *POINTS.write().unwrap() = contours
                        .iter()
                        .map(|contour| {
                            let moments = moments(&contour, false).unwrap();

                            // // Calculate area
                            // let area = contour_area(&contour, false).unwrap();

                            // Calculate bounding rectangle
                            let rect = bounding_rect(&contour).unwrap();
                            let size = rect.size();

                            Rect::from_center_size(
                                Pos2::new(
                                    (moments.m10 / moments.m00) as f32,
                                    (moments.m01 / moments.m00) as f32,
                                ),
                                Vec2::new(size.width as f32, size.height as f32),
                            )
                        })
                        .filter(|rect| rect.is_finite())
                        .collect::<Vec<_>>();
                }
            }
        });

        Self { image }
    }
}

impl eframe::App for CalibratorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_pixels_per_point(1.);

        Area::new("video feed")
            .fixed_pos(Pos2::ZERO)
            .show(ctx, |ui| {
                Image::new(&self.image)
                    .fit_to_exact_size(ui.available_size())
                    .maintain_aspect_ratio(true)
                    .paint_at(ui, Rect::from_min_size(Pos2::ZERO, ui.available_size()));

                for point in POINTS.read().unwrap().iter() {
                    ui.painter()
                        .rect_stroke(*point, 0., Stroke::new(1., Color32::RED))
                }
            });

        Window::new("Settings")
            .default_size([200.0, 200.0])
            .show(ctx, |ui| {
                let settings = unsafe { &mut SETTINGS };

                for (name, value, range) in [
                    ("lower_h", &mut settings.lower_h, 0.0..=180.0),
                    ("lower_s", &mut settings.lower_s, 0.0..=255.0),
                    ("lower_v", &mut settings.lower_v, 0.0..=255.0),
                    ("upper_h", &mut settings.upper_h, 0.0..=180.0),
                    ("upper_s", &mut settings.upper_s, 0.0..=255.0),
                    ("upper_v", &mut settings.upper_v, 0.0..=255.0),
                ] {
                    ui.add(
                        DragValue::new(value)
                            .clamp_range(range)
                            .speed(0.1)
                            .prefix(name),
                    );
                }
            });

        ctx.request_repaint();
    }
}
