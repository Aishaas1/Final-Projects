# اعتمادات المصادر والأوزان

ملفا `docs/sources/ffhq-dataset_README.md` و`stylegan_README.md` نسختان من توثيق NVIDIA الأصلي، جرى تنزيلهما دون تعديل في2026-11-06. الأول ضمن مادة FFHQ المرخصة CC BY-NC-SA4.0، والثاني من مادة StyleGAN المرخصة CC BY-NC4.0. الروابط والتفاصيل في بطاقة البيانات. ينبغي الحفاظ على هذه النسب وشروط المشاركة عند إعادة توزيع هذين الملفين. الشرح العربي يذكر التعديلات التي يجريها المشروع على الصور إذا نزّلها المستخدم.

MobileNetV3-Small مصدره torchvision/PyTorch؛ أوزان ImageNet الرسمية تُنزّل تلقائيًا من `https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth` مع تحقق التجزئة. [توثيق النموذج](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html) و[ترخيص torchvision](https://github.com/pytorch/vision/blob/main/LICENSE) و[شروط ImageNet](https://www.image-net.org/download.php). الأوزان المرفقة fine-tuned لاختبار برمجيات فقط، ولم تُدرَّب على بيانات FFHQ/Kaggle في هذه الجلسة.

OpenCV يوفّر Haar cascade للوجه الأمامي في حزمة opencv-python-headless، دون تنزيل كاشف إضافي. بوابة الوجه قد تخطئ وليست جزءًا من تقرير دقة التصنيف العلمي. تسري تراخيص المكتبات الأصلية على كل مكتبة؛ requirements لا ينقل ملكيتها إلى المشروع.
